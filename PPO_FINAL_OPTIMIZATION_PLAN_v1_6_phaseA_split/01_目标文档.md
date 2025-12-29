# PPO_project 目标文档（FINAL v1.6）
版本日期：2025-12-30  
面向：RCIM 投稿 + 可复现代码工件（Artifact）

---

## 1) 总目标（系统层面）
在轮廓误差容许带（tolerance band，±ε/2）约束下学习策略 π，使其在不同轨迹族上同时满足：

- **直线段**：高进给、贴线（误差不超带）
- **拐角段**：容许带内自动平滑（降低 jerk / 角加速度峰值），降速尽量少
- **全程**：到达终点、无停滞、运动学约束零违规（通过 KCM action shielding 保证）

---

## 2) 论文必须报告的指标（最小集合）
要求：所有指标由脚本生成，并落到 `artifacts/<phase>_accept/summary.json`；且支持 **straight / corner / exit** 分段统计。

### 成功与稳定性
- `success_rate`
- `stall_rate`
- `mean_progress_final`

### 精度（硬约束）
- `half_epsilon`
- `max_abs_e_n`（必须 ≤ `half_epsilon`）
- `rmse_e_n`

### 效率
- `cycle_time_steps_mean`
- `mean_v_ratio`
- `mean_v_ratio_straight`（“直线不退化”的核心）

### 平滑性（拐角重点）
- `peak_jerk_corner`（或 proxy）
- `peak_ang_acc_corner`（或 proxy）
- `roughness_proxy`

### 约束与干预（推荐但强烈建议）
- `kcm_violation_rate`（目标 0）
- `mean_kcm_intervention`（越低越好）

### 出弯专项（B2b 引入）
- `exit_recovery_steps_mean`
- `exit_oscillation_rms`

---

## 3) 论文主张边界（用词要严格）
- **7 维状态**：作为“最小充分统计量”的工程化设计（B1），用实验显示“更稳/更省样本/不退化”
- **KCM**：仅表述为 action shielding / feasibility projection（不进计算图，不写端到端可微）
- **平滑是学习出来的**：B2a/b/c 的改动是 reward/状态/软约束的可消融设计，不是规则生成轨迹

---

## 4) 通用硬约束（任何阶段不得破坏）
- `max_abs_e_n <= half_epsilon`
- `success_rate >= 0.80`
- `stall_rate <= 0.05`
- `mean_progress_final >= 0.95`
- `kcm_violation_rate == 0`（若暂未统计，必须在 Phase A-1 补齐统计）

