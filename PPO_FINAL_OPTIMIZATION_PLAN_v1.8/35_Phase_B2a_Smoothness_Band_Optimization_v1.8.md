# Phase B2a+：允差带内平滑优化方案（方案1+2+3）
版本日期：2026-01-04  
关联：`PPO_FINAL_OPTIMIZATION_PLAN_v1.8/30_Phase_B2a_FINAL_v1.8.md`

---

## 1) 理解阶段（现状与痛点）
- 轨迹仍贴中心线，拐角几何形态几乎不变，仅角加速度下降。
- 当前奖励函数强力惩罚 `contour_error` 与航向偏差，导致最优策略趋向“贴线+对齐切线”。
- 平滑项仅惩罚 jerk/角jerk，权重较小且只在 corner_mask 生效，无法改变路径几何。
- 目标是“允差带内尽可能平滑”，而非“严格贴中心线”。

---

## 2) 规划阶段（目标与可测指标）
**目标：**
- 在拐角阶段允许在允差带内“走平滑轨迹”，而非贴线；
- 直线段不退化（贴线与效率不变或更好）；
- 硬约束（越界/失败）不增加。

**可测指标（建议）：**
- `max_abs_contour_error` 仍 <= `half_epsilon`；
- corner_mask 段 smoothness 指标（omega/domega 或 jerk_proxy）下降 ≥ 10~20%；
- 直线段 `steps` 与 `progress_final` 不退化。

---

## 3) 执行阶段（方案1+2+3）

### 3.1 方案1：带内“平坦追踪” + 带外强惩罚（仅拐角）
**核心：** 在 corner_mask 段，允许 `e_n` 在允差带内“自由”以让平滑项主导；带外则强惩罚。

**新增/调整参数（reward_weights）：**
- `track_deadzone_ratio`（默认 0.25）：带内“平坦区”占比；
- `track_outside_weight`（默认 2.0）：带外额外惩罚倍率；
- 仅在 `corner_mask=True` 时生效，直线段维持原追踪惩罚（保持可归因）。

**逻辑示意（伪代码）：**
```
if corner_mask:
    dead = track_deadzone_ratio * half_epsilon
    if |e| <= dead: r_track = 0
    elif |e| <= half_epsilon: r_track = -w_e * ((|e|-dead)/(half_epsilon-dead))^2
    else: r_track = -w_e * (|e|/half_epsilon)^2 * track_outside_weight
else:
    r_track = -w_e * (|e|/half_epsilon)^2
```

### 3.2 方案2：拐角航向惩罚衰减
**核心：** 允许拐角轻微切角，不强制“对齐中心线切线”。

**新增参数（reward_weights）：**
- `corner_w_tau_scale`（默认 0.4~0.6）

**逻辑示意：**
```
w_tau_eff = w_tau * corner_w_tau_scale if corner_mask else w_tau
r_dir = -w_tau_eff * tau^2
```

### 3.3 方案3：加强平滑项（角加速度/角jerk）
**核心：** 在拐角阶段增加对 `angular_acc` 的惩罚（更直接改善“圆滑性”）。

**参数建议：**
- `w_smooth`：从 0.02 提升到 0.10~0.20；
- 新增 `w_ang_acc`（默认 0.05）：与 `max_ang_acc` 归一化。

**逻辑示意：**
```
if corner_mask:
    r_smooth = -w_smooth * (jerk_ratio^2 + ang_jerk_ratio^2)
    r_smooth += -w_ang_acc * (ang_acc/max_ang_acc)^2
```

---

## 4) 实施步骤（最小改动，保证可归因）
1. 修改 `PPO_project/src/environment/reward.py`：引入带内平坦追踪、corner 航向缩放、角加速度惩罚。
2. 复制配置：`PPO_project/configs/train_square_b2a.yaml` → `train_square_b2a_smooth.yaml`，仅改 `reward_weights`。
3. 短跑 100 episodes 验证：观察 `overlay.svg` 与 `trajectory_points.svg`（允差带内是否“切角更圆滑”）。
4. 若通过，再进行长跑与归档（A1/A3/plotter 与 `main_table.csv` 更新）。

---

## 5) 验收标准（PASS/FAIL）
**PASS（全部满足）：**
- `success_rate` ≥ 0.8；
- `max_abs_contour_error` ≤ `half_epsilon`；
- corner_mask smoothness 指标下降 ≥ 10~20%；
- 直线段进度与速度不退化（steps 不明显增加）。

**FAIL：**
- 越界/失败增多；
- 依靠降低全程速度获得“伪平滑”。

---

## 6) 风险与回滚策略
- **偏离过大**：降低 `track_deadzone_ratio` 或提高 `track_outside_weight`。
- **仍贴中心线**：提高 `track_deadzone_ratio` 或降低 `corner_w_tau_scale`。
- **拐角过慢**：降低 `w_smooth` 或提高 P4 的 `v_min`（仅 corner_mask）。

---

## 7) 原则体现（KISS / YAGNI / DRY / SOLID）
- **KISS**：仅调整 RewardCalculator，保持环境与训练流程不变。
- **YAGNI**：不引入新模块或新状态，只增加必要参数。
- **DRY**：追踪/平滑逻辑集中在 reward 内部，避免分散改动。
- **SRP（SOLID）**：RewardCalculator 仅负责奖励计算，其他逻辑不耦合。

---

## 8) 附录：推荐参数（起步值）
```
reward_weights:
  w_e: 5.0
  w_tau: 2.0
  w_smooth: 0.12
  w_ang_acc: 0.05
  smooth_corner_only: true
  track_deadzone_ratio: 0.25
  track_outside_weight: 2.0
  corner_w_tau_scale: 0.5
```
