# Phase 31：可学习前瞻距离（v4.0 执行稿）

版本日期：2026-02-27  
依赖：Phase 30（固定前瞻 + corridor/KCM 基线）  
目标：把 Phase31 主线从“仅前瞻观测”升级为“**前瞻距离可学习控制**”，实现：
- 直线段：短前瞻、紧密跟踪
- 拐角段：长前瞻、提前转向并抑制角速度尖峰
- 全程：KCM 约束安全

---

## 1. 核心假设

### H1：学习前瞻距离可恢复固定前瞻在拐角的平滑优势
固定前瞻在拐角有效，是因为它等效引入了提前转向；可学习前瞻若学到“拐角拉长、直线缩短”，应复现并超过该效果。

### H2：分区奖励能避免“全局折中”
将路径分为直线/拐角区域后，直线保持高跟踪梯度，拐角加强平滑项，可减少“直线漂、拐角尖”的冲突。

### H3：前瞻控制权必须进入策略动作
仅把 lookahead 放入观测，可能学不到“控制权”。将前瞻控制量 `u_lookahead∈[0,1]` 并入动作空间，策略才真正拥有可学习控制权。

---

## 2. 方案定义（Phase31 主线）

### 2.1 状态与动作
- 状态：保留多尺度 lookahead 观测（固定维度）
- 动作：`[theta_u, v_u, u_lookahead]`
  - `u_lookahead` 映射到 `[min_dist, max_dist]`

### 2.2 前瞻距离执行逻辑
- 基础距离：
  - 直线目标 `straight_dist`
  - 拐角目标 `corner_dist`
- 分区权重：`region_weight∈[0,1]`（由 turn/cornerness 提供）
- 合成：
  - `base_dist = straight_dist + region_weight * (corner_dist - straight_dist)`
  - `policy_dist = min_dist + u_lookahead * (max_dist - min_dist)`
  - `active_dist = (1-mix_gain)*base_dist + mix_gain*policy_dist`

### 2.3 分区奖励（直线/拐角）
- 跟踪/平滑权重由 cornerness 连续调度（已在 reward 中实现）
- 新增前瞻控制奖励：
  - 直线目标：`lookahead_dist_norm -> straight_target`
  - 拐角目标：`lookahead_dist_norm -> corner_target`
  - 按 `region_weight` 加权

### 2.4 主线约束
- `reward_weights.minimal_mode = false`
- `reward_weights.tube.enabled = false`
- `corridor.enabled = true`（保留几何检测）
- corridor 强惩罚默认降级为 0（减少规则主导）

---

## 3. 主配置（v31 mainline）

使用：`PPO_project/configs/train_square_v31_v3_2.yaml`

关键参数：
- `lookahead_control.enabled: true`
- `lookahead_control.policy_action: true`
- `lookahead_control.region_source: cornerness`
- `lookahead_control.min_dist: 0.8`
- `lookahead_control.max_dist: 4.0`
- `lookahead_control.straight_dist: 1.2`
- `lookahead_control.corner_dist: 2.8`
- `lookahead_control.mix_gain: 0.85`
- `lookahead_reward.enabled: true`
- `lookahead_reward.straight_target: 0.20`
- `lookahead_reward.corner_target: 0.80`

---

## 4. 验收标准（Phase31）

### MUST（硬门槛）
- `success_rate >= 0.95`
- `max_abs_e_n <= 0.30 * epsilon`
- `steps <= 1.2 * baseline`

### LEARNING-EFFECT（学习有效性门槛）
- 动作空间维度为 3（含 `u_lookahead`）
- `lookahead_dist_active` 在评估轨迹中非常数（有控制变化）
- 拐角区平均前瞻距离 > 直线区平均前瞻距离

### PERFORMANCE（效果门槛，vs 固定前瞻基线）
- `corner_peak_abs_omega` 下降（目标 >= 10%）
- 直线段平均误差不退化
- `progress_final` 不低于基线

---

## 5. 必做验证

### V1：控制权有效性（无需长训练）
脚本：`PPO_project/tools/phase31_learnable_validate.py`
检查：
- `u_lookahead` 是否影响 `lookahead_dist_active`
- 分区逻辑下是否满足“拐角长、直线短”

### V2：短训可学习性（建议 100~200 episode）
检查：
- `lookahead_dist_norm` 分布从单峰向分区结构演化
- `r_lookahead` 与 corner_phase 有对应关系

### V3：主训与对比（多 seed）
- 与 Phase30 固定前瞻基线做 A/B 对比
- 输出 compare_metrics + 轨迹图

---

## 6. 论文可用表述

我们将前瞻信息从固定启发式参数升级为策略可控量，将前瞻距离并入动作空间，并用几何分区奖励约束“直线短前瞻、拐角长前瞻”的行为先验；该设计同时保留端到端学习范式与工程可解释性。

---

## 7. 交付物
- 主配置：`train_square_v31_v3_2.yaml`
- 验证脚本：`tools/phase31_learnable_validate.py`
- 验证输出：`out/phase31_learnable_validate/summary.json`
- 对比输出：`out/phase31_cmp_ad/*`
