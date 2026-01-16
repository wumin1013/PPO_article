# v1.9 → v2.0 迁移指南
版本日期：2026-01-17

---

## 0) 为什么迁移？

### v1.9 的问题

1. **规则过多**：corridor, dir_pref, center_weight 等 15+ 参数需要调优
2. **调参困难**：参数间存在复杂交互，难以找到全局最优
3. **科研叙事弱**：本质是"规则工程"而非"端到端学习"
4. **B2a 实验失败**：反复调参但 corner_peak_omega 无改善

### v2.0 的优势

1. **极简设计**：只有 4 个奖励项
2. **可解释性强**：每个组件作用明确
3. **科研价值高**：如果成功，证明 RL 能自主学习平滑策略
4. **调参简单**：参数少，交互简单

---

## 1) 保留的工作

以下 v1.9 的成果**完全保留**，无需重做：

| Phase | 产物 | 说明 |
|-------|------|------|
| Phase 20 | 代码清理 | Cleanup 是正确的 |
| Phase 21 | 12 维状态 | 状态精简有效 |
| Phase 22 | P0 基线 | 基线模型可用 |
| Phase 23 | P0_L2 归档 | 作为对比基线 |

**基线路径**：`artifacts/P0_L2/P0_12d_gold_20260114_174552/`

---

## 2) 废弃的工作

以下 v1.9 的成果**废弃**：

| Phase | 说明 | 原因 |
|-------|------|------|
| Phase 30 (B2a) | corridor 方案 | 技术路线错误 |
| B2a 实验 | `artifacts/B2a/*` | 可删除或归档 |
| 相关配置 | `train_square_b2a*.yaml` | 保留作参考 |

### 清理命令（可选）

```powershell
# 归档 B2a 实验（不删除，留作对比）
Move-Item -Path "artifacts/B2a" -Destination "artifacts/_archived_b2a_v1.9"

# 或直接删除
# Remove-Item -Recurse -Path "artifacts/B2a"
```

---

## 3) 代码修改清单

### 3.1 `src/environment/reward.py`

**修改内容**：添加 minimal reward 模式

```python
# 在 RewardCalculator 类中添加

def calculate_reward(self, ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
    """统一入口：根据配置选择奖励模式"""
    if self.weights.get("minimal_mode", False):
        return self._calculate_minimal_reward(ctx)
    else:
        return self._calculate_legacy_reward(ctx)

def _calculate_minimal_reward(self, ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
    """v2.0 极简奖励"""
    w_s = float(self.weights.get("w_s", 20.0))
    p4_cfg = self.weights.get("p4", {})
    time_penalty = float(p4_cfg.get("time_penalty", -0.02))
    stall_penalty = float(p4_cfg.get("stall_penalty", -8.0))
    boundary_cfg = self.weights.get("boundary", {})
    completion_cfg = self.weights.get("completion", {})
    
    # 1. Progress
    progress_diff = max(0.0, float(ctx.progress) - float(self.last_progress))
    r_progress = w_s * progress_diff
    
    # 2. Boundary
    r_boundary = 0.0
    if boundary_cfg.get("enabled", False):
        if abs(float(ctx.contour_error)) > float(self.half_epsilon):
            r_boundary = float(boundary_cfg.get("penalty", -100.0))
    
    # 3. Time
    r_time = time_penalty
    
    # 4. Completion
    r_completion = 0.0
    if completion_cfg.get("enabled", False) and bool(ctx.lap_completed):
        r_completion = float(completion_cfg.get("reward", 50.0))
    
    # 5. Stall
    r_stall = 0.0
    if bool(ctx.stall_triggered):
        r_stall = stall_penalty
    
    total = r_progress + r_boundary + r_time + r_completion + r_stall
    self.last_progress = float(ctx.progress)
    
    return total, {
        "r_progress": r_progress,
        "r_boundary": r_boundary,
        "r_time": r_time,
        "r_completion": r_completion,
        "r_stall": r_stall,
        "total": total,
    }

def _calculate_legacy_reward(self, ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
    """原 v1.9 奖励逻辑"""
    # 将原 calculate_reward 的内容移到这里
    ...
```

### 3.2 配置文件

**创建**：`configs/train_square_minimal.yaml`

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
  MAX_ANG_VEL: 6.283185307179586
  MAX_ANG_ACC: 100.0
  MAX_ANG_JERK: 1000.0

reward_weights:
  minimal_mode: true  # 启用 v2.0 极简模式
  w_s: 20.0
  w_e: 0.0    # 禁用
  w_tau: 0.0  # 禁用
  w_smooth: 0.0
  
  boundary:
    enabled: true
    penalty: -100.0
  
  completion:
    enabled: true
    reward: 50.0
  
  corridor:
    enabled: false  # 禁用 corridor
  
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
  name: minimal_v1
  enable_kcm: true
```

---

## 4) 迁移步骤

### Step 1：备份（2 分钟）

```powershell
# 备份 B2a 实验
Move-Item "artifacts/B2a" "artifacts/_archived_b2a_v1.9"
```

### Step 2：代码修改（30 分钟）

1. 修改 `src/environment/reward.py`
2. 创建 `configs/train_square_minimal.yaml`
3. 验证配置加载

### Step 3：验证（5 分钟）

```powershell
# 快速验证配置
python -c "
import yaml
with open('configs/train_square_minimal.yaml') as f:
    cfg = yaml.safe_load(f)
print('minimal_mode:', cfg['reward_weights'].get('minimal_mode'))
print('corridor.enabled:', cfg['reward_weights']['corridor'].get('enabled'))
"
# 期望：minimal_mode: True, corridor.enabled: False
```

### Step 4：开始 Phase 30

```powershell
python main.py --config configs/train_square_minimal.yaml --mode train
```

---

## 5) 文档对照表

| v1.9 文档 | v2.0 对应 | 状态 |
|-----------|-----------|------|
| 01_Objectives_FINAL_v1.9.md | 01_Objectives_v2.0.md | 替换 |
| 02_Workflow_FINAL_v1.9.md | 02_Workflow_v2.0.md | 替换 |
| 20_Phase_Cleanup*.md | 保留 | 不变 |
| 21_Phase_StateSpace*.md | 保留 | 不变 |
| 22_Phase_P0*.md | 保留 | 不变 |
| 23_Phase_P0_L2*.md | 保留 | 不变 |
| 30_Phase_B2a*.md | 30_Phase_Minimal*.md | 替换 |
| 40_Phase_B2b*.md | 废弃 | - |
| 50_Phase_B2c*.md | 废弃 | - |
| 60_Phase_C*.md | 废弃 | - |
| 70_Phase_D*.md | 废弃 | - |
| 90_Swimlane*.md | 90_Swimlane_v2.0.md | 替换 |

---

## 6) 常见问题

### Q1: 为什么不从头重训 P0 基线？

Phase 20-23 的工作是正确的：
- 代码清理提高了可维护性
- 12 维状态空间是合理的精简
- P0_L2 基线模型有效

问题出在 Phase 30 的技术路线（规则过多），不是基础设施。

### Q2: 如果 v2.0 极简方案也失败怎么办？

按优先级尝试：
1. 增加 time_penalty
2. 延长训练
3. 加入曲率状态（Phase 31）
4. 加入极轻量曲率惩罚

最坏情况：仍比 v1.9 简单得多（1 个惩罚 vs 15+ 个）

### Q3: B2a 实验数据要删除吗？

建议归档而非删除：
- 可作为"规则工程 vs 端到端"的对比案例
- 论文 related work / discussion 可能引用

### Q4: 配置文件中保留的参数有什么用？

保留但设为禁用（如 `w_e: 0.0`）的参数是为了：
- 代码兼容性
- 方便快速回滚对比
- 清晰展示"我们禁用了什么"

---

## 7) 快速开始

如果你刚读到这里，想立即开始：

```powershell
# 1. 进入项目目录
cd PPO_project
conda activate PPO

# 2. 阅读 Phase 30 文档
# PPO_OPTIMIZATION_PLAN_v2.0/30_Phase_Minimal_Reward_v2.0.md

# 3. 实施代码修改（按文档说明）

# 4. 运行训练
python main.py --config configs/train_square_minimal.yaml --mode train

# 5. 评估
python tools/a1_pack_run.py --run_dir artifacts/minimal_v1 --baseline artifacts/P0_L2/P0_12d_gold_20260114_174552
```
