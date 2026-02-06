# Phase 31：前瞻作为观测（v3.1 观测引导方案）
版本日期：2026-01-28  
依赖：Phase 30 完成（v3.0 基线可用）

---

## 0) 问题陈述

### Phase 30 的学术缺陷

当前 v3.0 设计将**前瞻方向直接作为参考方向 θ_ref**：

```python
# v3.0 问题代码：
theta_lookahead = self._get_path_direction(position)  # 前瞻方向
heading_error = theta_current - theta_lookahead       # 直接作为参考
r_direction = -w * abs(heading_error)                 # 惩罚偏离
```

**问题**：
- 这本质上是**人工编码的"如何平滑过弯"知识**
- 审稿人可能质疑：*这不就是经典的 Pure Pursuit 算法吗？*
- **RL 贡献不清晰**：到底是学习到的还是工程配置出来的？

---

## 1) 核心改进（v3.1）

### 关键转变：前瞻从"指令"变为"观测"

```diff
- v3.0: 前瞻方向 → 直接作为参考方向 θ_ref → 奖励计算惩罚偏离
+ v3.1: 前瞻信息 → 作为状态观测 → 策略自主决定是否/如何利用
```

### 稳定落地建议（护栏版，推荐执行顺序）

你当前工程基线已经很强（PPO + KCM + corridor/turn_info + fixed lookahead LOS），因此 v3.1 的落地目标不是“推倒重来”，而是**在不破坏稳定性的前提下**逐步去除人工规则依赖，让贡献归因更清晰、论文叙事更硬。

**推荐执行顺序（1→4）**
1. **恢复前瞻观测**（lookahead as observation），但保持**观测维度固定**，便于做消融与复现实验。
2. **保留 corridor 做检测上下文**（cornerness / turn_info），先把 corridor 的“强惩罚/强引导”降级（权重→0），避免一刀切导致训练早期发散。
3. 引入 **cornerness 自适应权重**：弯道更偏平滑，直线更偏精度（权重连续变化 + 下限/上限）。
4. 做消融并固化结论：`corridor(惩罚) on/off`、`cornerness-adaptive on/off`、`lookahead-obs on/off`。

---

### 护栏 A：corridor 不要删，先“降级为检测”

当前代码里 `corridor.enabled` 同时承担了两类职责：
- **检测/上下文**：提供 `corner_phase / turn_sign / dist_to_turn ...`（用于 state/日志/可解释性）。
- **奖励塑形**：通过 outside/barrier/center/dir_pref 等项“硬拉”策略。

论文路线建议：先把（2）逐步退出，而不是砍掉（1）。

**最小改动方案（优先推荐）**：保持 `reward_weights.corridor.enabled: true`，但将塑形权重置 0（仍保留 corner 检测与日志字段）。
- `outside_penalty_weight: 0.0`
- `barrier_weight: 0.0`
- `center_weight: 0.0`
- `dir_pref_weight: 0.0`

这样做的好处：训练稳定性不掉（仍有上下文），同时避免审稿人质疑“靠手工规则惩罚跑出来”。

---

### 护栏 B：cornerness 信号必须“归一化 + 裁剪 + 平滑”

方形路径的拐角曲率本质是冲击，直接用数值曲率容易抖动，PPO 很容易不稳定。推荐用**前瞻角度变化**构造 cornerness（更鲁棒）。

建议定义（示意）：
- `c_raw = |wrap(theta_far - theta_now)| / theta0`
- `c = clip(EMA(c_raw), 0, 1)`

参数建议：
- `theta0` 取 20°~45°（代表“明显需要转弯”的归一化阈值）
- EMA 时间常数 5~20 steps（按你的 `dt` 与角速度动态调）

并设置权重上下界（防止“弯道过度放松导致飞出管道/边界”）：
- 跟踪惩罚 `w_e_eff >= w_e_floor`
- 平滑惩罚 `w_smooth_eff <= w_smooth_cap`

---

### 设计原理

| 维度 | v3.0 (指令引导) | v3.1 (观测引导) |
|------|-----------------|-----------------|
| **参考方向** | 前瞻向量（人工规定） | **路径切线**（客观几何） |
| **前瞻信息** | 隐式用于计算 θ_ref | **显式作为状态观测** |
| **策略自由度** | 被"强制"跟随前瞻 | 自主决定如何利用前瞻信息 |
| **学术定位** | 工程优化 | **自主学习** |

### 论文话术

> "We provide lookahead information as **observation features** rather than as an imperative reference direction. The agent autonomously learns **when** and **how** to utilize this predictive information, discovering smooth cornering strategies through reinforcement learning without explicit path-following heuristics."

---

## 2) 状态空间扩展（固定维度，支持消融）

### 原则：观测维度固定，关闭时置 0（而不是改网络结构）

为了让消融实验更干净（不换网络结构、不换 checkpoint 格式），建议将 lookahead 观测**固定为 N 个尺度**（推荐 3 个：near/mid/far），当 `lookahead_obs=false` 时将这些特征置 0。

推荐尺度（示例）：以 `lookahead.distance` 为 base，设置 `obs_scales=[0.5, 1.0, 2.0]`，分别对应近/中/远的前瞻信息。

### 新增观测特征（示例：每个尺度 2 维）

```python
# 示例：3 个尺度 → 6 维前瞻观测（添加到 _build_state 末尾）
lookahead_keys = [
    "la1_angle_diff", "la1_dist_ratio",
    "la2_angle_diff", "la2_dist_ratio",
    "la3_angle_diff", "la3_dist_ratio",
]
```

### 特征计算逻辑

```python
def _compute_lookahead_observation(self) -> Tuple[float, float]:
    """v3.1: 计算单尺度前瞻观测特征（而非参考方向）"""
    # 1. 获取当前位置的路径切线方向（客观几何参考）
    s_current = self._get_current_arc_length()
    theta_tangent = self._tangent_angle_at_s(s_current)
    
    # 2. 获取前瞻点和前瞻方向
    lookahead_dist = float(getattr(self, "lookahead_dist", 4.5))
    s_target = s_current + lookahead_dist
    target_point, _, _ = self._interpolate_point_at_s(s_target)
    
    dx = target_point[0] - self.current_position[0]
    dy = target_point[1] - self.current_position[1]
    dist_to_target = math.sqrt(dx*dx + dy*dy)
    
    if dist_to_target < 1e-9:
        return 0.0, 1.0  # 到达前瞻点，无角度差
    
    theta_lookahead = math.atan2(dy, dx)
    
    # 3. 计算前瞻与切线的角度差（策略需要学习这个信号的意义）
    angle_diff = self._normalize_angle(theta_lookahead - theta_tangent)
    angle_diff_norm = angle_diff / math.pi  # 归一化到 [-1, 1]
    
    # 4. 距离比例（接近前瞻点时趋向0）
    dist_ratio = min(dist_to_target / lookahead_dist, 1.0)
    
    return float(angle_diff_norm), float(dist_ratio)
```

### 特征解读

| 特征 | 范围 | 含义 |
|------|------|------|
| `lookahead_angle_diff` | [-1, 1] | 前瞻方向与切线的偏差（正=需左转，负=需右转） |
| `lookahead_dist_ratio` | [0, 1] | 到前瞻点的相对距离（接近1=远，接近0=近） |

**关键**：策略需要**学习**如何解读这些观测：
- 直线段：`angle_diff ≈ 0` → 无需提前调整
- 接近拐角：`angle_diff ≠ 0` → 信号来了，但策略**自主决定**是否/如何响应

> 说明：如果采用多尺度观测（推荐），你会得到 near/mid/far 三组 `angle_diff/dist_ratio`。论文可解释为一种“几何注意力”的输入基础：直线更关注 near，拐角前更关注 far。

---

## 3) 奖励函数修改

### 关键改变：参考方向回归切线

```python
def _calculate_minimal_reward(self, ctx: RewardContext) -> Tuple[float, Dict]:
    """v3.1: 管道奖励 + 切线参考（非前瞻）"""
    
    # ❌ v3.0 错误做法：用前瞻方向作为参考
    # theta_ref = self._get_path_direction(position)  # 前瞻
    
    # ✅ v3.1 正确做法：用切线作为参考（客观几何）
    # heading_error 由 calculate_direction_deviation 计算，已使用切线
    
    # 其他奖励保持不变（progress, tube, boundary, time, completion）
    ...
```

**效果**：奖励函数不直接使用前瞻信息，策略需要从观测中**自主学习**利用前瞻信号的价值。

### v3.1（护栏版）建议：cornerness 自适应权重（连续分段，而非硬阈值）

目标：直线段更重视跟踪精度，拐角段更重视平滑（同时保留跟踪惩罚下限，避免飞出边界）。

建议按 `cornerness c∈[0,1]` 连续插值（示意）：
- `w_e_eff = lerp(w_e_straight, w_e_corner, c)` 且 `w_e_eff >= w_e_floor`
- `w_smooth_eff = lerp(w_smooth_straight, w_smooth_corner, c)` 且 `w_smooth_eff <= w_smooth_cap`

说明：
- `cornerness` 推荐来自“前瞻角度变化”的平滑信号（见护栏 B），而不是直接数值曲率。
- corridor 可以先保留用于检测/可解释性；塑形惩罚先置 0，避免“参数工程”质疑。

---

## 4) 代码修改清单

### 4.1 修改 `src/environment/cnc_env.py`

#### 添加配置参数（`__init__`）

```python
# v3.1: 前瞻观测配置
lookahead_cfg = self.reward_weights.get("lookahead", {})
self.lookahead_enabled = bool(lookahead_cfg.get("enabled", True))
self.lookahead_dist = float(lookahead_cfg.get("distance", 4.5))
self.lookahead_as_observation = bool(lookahead_cfg.get("as_observation", True))  # v3.1 新增
```

#### 添加前瞻观测计算

```python
def _compute_lookahead_observation(self) -> Tuple[float, float]:
    """v3.1: 前瞻观测特征"""
    if not getattr(self, "lookahead_enabled", False):
        return 0.0, 1.0
    
    s_current = self._get_current_arc_length()
    theta_tangent = self._tangent_angle_at_s(s_current)
    
    lookahead_dist = float(getattr(self, "lookahead_dist", 4.5))
    s_target = s_current + lookahead_dist
    target_point, _, _ = self._interpolate_point_at_s(s_target)
    
    dx = target_point[0] - self.current_position[0]
    dy = target_point[1] - self.current_position[1]
    dist_to_target = math.sqrt(dx*dx + dy*dy)
    
    if dist_to_target < 1e-9:
        return 0.0, 1.0
    
    theta_lookahead = math.atan2(dy, dx)
    angle_diff = self._normalize_angle(theta_lookahead - theta_tangent)
    angle_diff_norm = float(angle_diff / math.pi)
    dist_ratio = float(min(dist_to_target / lookahead_dist, 1.0))
    
    return angle_diff_norm, dist_ratio
```

#### 修改 `_build_state`

```python
def _build_state(self) -> np.ndarray:
    """Phase 31: 构建14/16维状态向量（含前瞻观测）"""
    # ... 原有 12 维核心+拐角特征 ...
    
    # 2维曲率特征（可选）
    if self.enable_curvature_obs:
        kappa_norm, dkappa_norm = self.compute_curvature_features()
        state_values.extend([kappa_norm, dkappa_norm])
    
    # v3.1: 2维前瞻观测特征
    if getattr(self, "lookahead_as_observation", False):
        angle_diff_norm, dist_ratio = self._compute_lookahead_observation()
        state_values.extend([angle_diff_norm, dist_ratio])
    
    return np.array(state_values, dtype=np.float32)
```

#### 修改 `_get_path_direction`

```python
def _get_path_direction(self, pt, v_exec=None, record=False):
    """v3.1: 始终返回切线方向（前瞻仅作为观测）"""
    if getattr(self, "lookahead_as_observation", False):
        # v3.1: 回归切线参考（让观测引导，而非指令引导）
        return self._get_path_direction_legacy(pt, v_exec=v_exec, record=record)
    
    # v3.0 兼容：前瞻作为参考（用于对比实验）
    if getattr(self, "lookahead_enabled", False):
        ... # 原有前瞻逻辑
```

### 4.2 修改观测空间维度

```python
# __init__ 中
if self.lookahead_as_observation:
    # 每个尺度 2 维（angle_diff, dist_ratio）
    # obs_dim = base_dim(12/14) + 2 * N_scales
    self.observation_dim += 2 * N_scales  # 例如 N_scales=3: 12→18 / 14→20
```

---

## 5) 配置文件

### 创建 `configs/train_square_v31.yaml`

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
  MAX_ANG_VEL: 31.41592653589793
  MAX_ANG_ACC: 500.0
  MAX_ANG_JERK: 5000.0

reward_weights:
  minimal_mode: true
  w_s: 20.0
  w_e: 0.0
  w_tau: 0.0
  w_smooth: 0.0
  
  # v3.1: 前瞻作为观测
  lookahead:
    enabled: true
    distance: 4.5
    as_observation: true   # ← 关键：前瞻作为观测而非指令
  
  # 管道配置（保持不变）
  tube:
    enabled: true
    ratio: 0.5
  
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
  name: v31_lookahead_obs
  enable_kcm: true
```

---

## 6) 执行步骤

### Step 1：代码修改（30分钟）
1. 修改 `cnc_env.py`：添加 `_compute_lookahead_observation`、更新 `_build_state`、更新观测空间维度
2. 修改 `_get_path_direction`：`as_observation=True` 时回归切线
3. 创建 `configs/train_square_v31.yaml`

### Step 2：Smoke Test（5分钟）
```powershell
conda activate PPO
cd PPO_project
# 将 configs/train_square_v31.yaml 里 training.num_episodes 临时改小（例如 5），再运行训练入口
python main.py --config configs/train_square_v31.yaml --mode train
```

验证：
- `env.observation_space.shape[0] == base_dim(12/14) + 2 * N_scales`
- `la*_angle_diff` 在拐角前非零、直线段接近零（near/mid/far 任一尺度可观测）
- 奖励使用切线参考（非前瞻）

### Step 3：训练（2-3小时）
```powershell
python main.py --config configs/train_square_v31.yaml --mode train
```

### Step 4：评估
```powershell
python tools/acceptance_suite.py `
    --phase p0_eval `
    --config configs/train_square_v31.yaml `
    --model saved_models/v31_lookahead_obs/*/checkpoints/best_model.pth `
    --episodes 50 `
    --out out/v31_eval `
    --deterministic
```

---

## 7) 验收标准

### 必须项（MUST）

| 指标 | 条件 |
|------|------|
| success_rate | ≥ 0.95 |
| max_abs_e_n | ≤ 0.75mm |
| observation_dim | `base_dim(12/14) + 2 * N_scales` |

### 涌现项（OBSERVE）

| 指标 | 期望 | 解读 |
|------|------|------|
| corner_peak_omega | < 0.85 × MAX | 平滑涌现（策略学会利用前瞻信号） |
| inside_ratio | > 0.5 | 内切涌现 |
| lookahead_utilization | 可视化分析 | 策略在拐角前是否提前调整 |

---

## 8) 消融对比

### Phase 33 扩展

| 实验 | 配置 | 目的 |
|------|------|------|
| A | `as_observation: false` | v3.0 复现（前瞻作为指令） |
| B | `enabled: false` | 无前瞻（纯切线） |
| **C** | `as_observation: true` | **v3.1 主方案** |

**预期结论**：C 与 A 性能相当，但 C 的学术定位更清晰（自主学习 vs 工程配置）。

---

## 9) 学术价值

### v3.1 相比 v3.0 的优势

| 维度 | v3.0 | v3.1 |
|------|------|------|
| **创新性** | 工程技巧（Pure Pursuit） | 状态设计 + 自主学习 |
| **可解释性** | 策略"被迫"跟随 | 策略"主动选择" |
| **论文话术** | "We use lookahead..." | "We observe how the agent learns to utilize..." |

### 论文定位更新

> "Rather than prescribing lookahead as the reference direction, we provide it as **state observations**, enabling the agent to autonomously discover **when** to anticipate turns and **how** to adjust its behavior proactively. This design preserves the RL paradigm of learning from experience while still benefiting from predictive geometric information."

---

## 10) 交付物

| 文件 | 说明 |
|------|------|
| `configs/train_square_v31.yaml` | 配置 |
| `artifacts/v31_lookahead_obs/` | Run Bundle |
| `artifacts/v31_lookahead_obs/summary.json` | 指标汇总 |
| 消融对比图 | v3.0 vs v3.1 vs 无前瞻 |
