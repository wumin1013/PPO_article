# Phase 30：极简奖励实验（Minimal Reward）
版本日期：2026-01-17  
依赖：Phase 23 完成（P0_L2 基线可用）

---

## 0) 目标（一句话）

**用极简奖励函数验证"效率压力驱动平滑"假设**：只给进度奖励和时间惩罚，让策略自己发现平滑过弯是效率最优解。

---

## 1) 核心假设

```
急转弯 → 高角速度 → KCM 限制线速度 → 位移减少 → 需要更多步数 → 累积时间惩罚
                                                                    ↓
平滑过弯 ← 策略优化 ← 最小化惩罚 ← 发现"平滑=高效" ←─────────────────┘
```

**我们不告诉策略如何过弯，只给它效率压力，看它能否自己学会。**

---

## 2) 奖励函数实现

### 2.1 极简公式

```python
def calculate_minimal_reward(ctx) -> float:
    """v2.0 极简奖励：只有进度、边界、时间"""
    
    # 1. 进度奖励（唯一正向激励）
    r_progress = 20.0 * progress_diff
    
    # 2. 边界惩罚（硬约束）
    if abs(ctx.e_n) > half_epsilon:
        r_boundary = -100.0
    else:
        r_boundary = 0.0
    
    # 3. 时间惩罚（效率压力）
    r_time = -0.02  # 略高于 v1.9 的 -0.01，增加效率压力
    
    # 4. 完成奖励（可选）
    r_completion = 50.0 if ctx.reached_target else 0.0
    
    return r_progress + r_boundary + r_time + r_completion
```

### 2.2 与 v1.9 P0 对比

| 奖励项 | v1.9 P0 | v2.0 Minimal |
|--------|---------|--------------|
| r_progress | ✅ w_s=20 | ✅ w_s=20 |
| r_track | ✅ w_e=5 | ❌ 删除 |
| r_dir | ✅ w_tau=2 | ❌ 删除 |
| r_time | ✅ -0.01 | ✅ -0.02 (增强) |
| r_smooth | ⚪ 可选 | ❌ 删除 |
| r_boundary | ❌ 无 | ✅ 新增（硬约束） |
| corridor | ⚪ B2a 用 | ❌ 删除 |

### 2.3 关键洞察

**为什么删除 r_track（贴线惩罚）？**
- 贴线惩罚强迫策略贴近中心线
- 但拐角最优轨迹可能需要偏离中心线（内切）
- r_boundary 已保证不越带，带内应自由

**为什么删除 r_dir（航向惩罚）？**
- 航向惩罚强迫策略与路径切线对齐
- 但平滑过弯时，航向自然会偏离局部切线
- 这是正常的几何特性，不应惩罚

---

## 3) 配置文件

### 3.1 创建 `configs/train_square_minimal.yaml`

```yaml
seed: 42

environment:
  epsilon: 1.5
  interpolation_period: 0.001
  max_steps: 4000
  lookahead_points: 0  # Phase 21: 禁用

kinematic_constraints:
  MAX_VEL: 100.0
  MAX_ACC: 2000.0
  MAX_JERK: 20000.0
  MAX_ANG_VEL: 6.283185307179586
  MAX_ANG_ACC: 100.0
  MAX_ANG_JERK: 1000.0

reward_weights:
  # v2.0 极简奖励
  w_s: 20.0           # 进度奖励
  w_e: 0.0            # 删除贴线惩罚
  w_tau: 0.0          # 删除航向惩罚
  w_smooth: 0.0       # 删除平滑惩罚
  
  # 边界惩罚（新增）
  boundary:
    enabled: true
    penalty: -100.0    # 越带硬惩罚
  
  # 完成奖励（新增）
  completion:
    enabled: true
    reward: 50.0
  
  # 禁用 corridor
  corridor:
    enabled: false
  
  # P4 时间惩罚
  p4:
    time_penalty: -0.02  # 增强效率压力
    stall_enabled: true
    stall_steps: 300
    stall_penalty: -8.0
  
  # 禁用其他 legacy
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
  num_episodes: 200  # 快验
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
  name: minimal_v1
  enable_kcm: true
```

---

## 4) 代码修改清单

### 4.1 `src/environment/reward.py`

添加 minimal reward 模式：

```python
class RewardCalculator:
    def calculate_reward(self, ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
        # 检查是否启用 minimal 模式
        if self.weights.get("minimal_mode", False):
            return self._calculate_minimal_reward(ctx)
        
        # 否则走原有逻辑
        return self._calculate_legacy_reward(ctx)
    
    def _calculate_minimal_reward(self, ctx) -> Tuple[float, Dict[str, float]]:
        """v2.0 极简奖励"""
        w_s = float(self.weights.get("w_s", 20.0))
        time_penalty = float(self.weights.get("p4", {}).get("time_penalty", -0.02))
        boundary_cfg = self.weights.get("boundary", {})
        completion_cfg = self.weights.get("completion", {})
        
        # 1. 进度
        progress_diff = max(0.0, ctx.progress - self.last_progress)
        r_progress = w_s * progress_diff
        
        # 2. 边界
        r_boundary = 0.0
        if boundary_cfg.get("enabled", False):
            if abs(ctx.contour_error) > self.half_epsilon:
                r_boundary = float(boundary_cfg.get("penalty", -100.0))
        
        # 3. 时间
        r_time = time_penalty
        
        # 4. 完成
        r_completion = 0.0
        if completion_cfg.get("enabled", False) and ctx.lap_completed:
            r_completion = float(completion_cfg.get("reward", 50.0))
        
        # 5. 停滞（保留，防止策略卡住）
        r_stall = 0.0
        if ctx.stall_triggered:
            r_stall = float(self.weights.get("p4", {}).get("stall_penalty", -8.0))
        
        total = r_progress + r_boundary + r_time + r_completion + r_stall
        self.last_progress = ctx.progress
        
        return total, {
            "r_progress": r_progress,
            "r_boundary": r_boundary,
            "r_time": r_time,
            "r_completion": r_completion,
            "r_stall": r_stall,
            "total": total,
        }
```

### 4.2 配置开关

在 yaml 中添加：
```yaml
reward_weights:
  minimal_mode: true  # 启用 v2.0 极简模式
```

---

## 5) 执行步骤

### Step 1：代码修改（30 分钟）
1. 修改 `reward.py`，添加 `_calculate_minimal_reward`
2. 创建 `configs/train_square_minimal.yaml`
3. 验证配置加载正确

### Step 2：快速训练（1-2 小时）
```powershell
conda activate PPO
cd PPO_project
python main.py --config configs/train_square_minimal.yaml --mode train
```

### Step 3：评估
```powershell
python tools/a1_pack_run.py --run_dir artifacts/minimal_v1 --baseline artifacts/P0_L2/P0_12d_gold_20260114_174552
python tools/rollout_trace.py --model artifacts/minimal_v1/checkpoint.pth --config configs/train_square_minimal.yaml --out artifacts/minimal_v1/rollout_det
```

### Step 4：涌现分析
```powershell
python tools/b2a1_corner_evidence.py --candidate artifacts/minimal_v1 --baseline artifacts/P0_L2/P0_12d_gold_20260114_174552
```

---

## 6) 验收标准

### 6.1 必须项（MUST）

| 指标 | 条件 |
|------|------|
| success_rate | ≥ 0.95 |
| max_abs_e_n | ≤ ε/2 = 0.75 |
| steps | ≤ 1.5× baseline（初期放宽） |

### 6.2 涌现指标（OBSERVE，不作为 PASS/FAIL）

| 指标 | 期望 | 解读 |
|------|------|------|
| corner_peak_omega | < 0.9 × MAX_ANG_VEL | 平滑涌现 |
| corner_min_v | > baseline | 高速过弯 |
| inside_ratio | > 0.5 | 内切涌现（bonus） |

---

## 7) 结果处理

### 7.1 如果涌现成功
```
corner_peak_omega 显著下降（相对 P0_L2）
→ ✅ 假设验证，进入 Phase 32（多路径验证）
```

### 7.2 如果未涌现但必须项通过
```
corner_peak_omega 无变化，但 success_rate 高
→ 尝试 Step A：增加 time_penalty（-0.02 → -0.05）
→ 尝试 Step B：延长训练（200 → 500 episodes）
→ 如果仍无变化 → 进入 Phase 31（曲率状态）
```

### 7.3 如果必须项失败
```
success_rate < 0.95 或 max_abs_e_n > ε/2
→ 检查 boundary penalty 是否过严
→ 考虑改用 soft boundary（梯度惩罚而非硬切）
```

---

## 8) 调优旋钮（备用）

| 旋钮 | 默认值 | 调整方向 | 效果 |
|------|--------|----------|------|
| `time_penalty` | -0.02 | ↓ 到 -0.05 | 增加效率压力 |
| `boundary.penalty` | -100 | ↑ 到 -50 | 减轻越带恐惧 |
| `w_s` | 20 | ↑ 到 30 | 增加进度激励 |
| `num_episodes` | 200 | ↑ 到 500 | 更多学习时间 |
| `ent_coef` | 0.01 | ↑ 到 0.02 | 增加探索 |

---

## 9) 交付物

| 文件 | 说明 |
|------|------|
| `configs/train_square_minimal.yaml` | 配置 |
| `artifacts/minimal_v1/` | Run Bundle |
| `artifacts/minimal_v1/summary.json` | 指标汇总 |
| `artifacts/minimal_v1/emergence_report.md` | 涌现分析报告 |

---

## 10) 论文映射

如果 Phase 30 成功涌现，可以在论文中强调：

> "本文采用极简奖励设计（仅包含进度激励和效率惩罚），策略在没有显式几何规则引导的情况下，**自主学习**了拐角平滑过渡策略。这验证了强化学习在运动规划任务中的端到端学习能力。"

这比"我们设计了15个奖励项来引导策略..."的叙事更有科研价值。
