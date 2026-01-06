# Phase B2a1：Arc-in-band 内切圆弧学习（符号对齐 + 走廊奖励生效）（FINAL v1.9）
版本日期：2026-01-06  
插入位置：`30_Phase_B2a_FINAL` 之后、`35_Phase_B2a_Smoothness...` 之前  
依赖：B1 已通过（主表列齐全、trace/summary/plotter 工作流稳定）

---

## 0) 这份 Phase 解决什么（一句话）
把“拐角内切圆弧”从口号变成可学习目标：  
**(A) 反向偏离先被消灭（符号/口径一致）**，  
**(B) 走廊/内切/航向偏好这些信号真的进入 reward**，  
从而让策略在允差带内稳定形成 **切线连续的圆弧过渡**，减少降速。

---

## 1) 先验判断：尖角为什么顽固
如果你只做了 B2a 的 deadzone + 平滑惩罚，但没有“几何 shaping”，策略大概率会选：
- “贴中线到最后一刻再转向”（几何上就是尖角），或
- “大幅降速换稳定”（效率退化）

要让它愿意提前转向并形成圆弧，需要两类驱动力：
1) **内侧偏好（inside bias）**：在不越带前提下，鼓励落在拐角内侧半带并保持一定深度；
2) **lookahead 航向引导**：鼓励航向与 lookahead 方向一致，从而提前、渐进地转向（切线连续）。

---

## 2) Step 0：10 分钟快检（不通过就不要训练）
### 0.1 快检 A：Zero Action Policy（reference 尖不尖？）
跑一条 action=0 的 trace（只跟随参考/残差为零）并看拐角 overlay：
- 若仍尖角：说明“指导航向/参考几何”本身是折线  
  → B2a1 必须启用 **lookahead 航向（theta_los）** 参与 reward/obs；否则 RL 很难学出提前转向。
- 若不尖角：说明尖角来自 reward/探索权衡  
  → 重点做 Step 2/3（内切偏好 + 切线连续奖励）。

### 0.2 快检 B：inside_ratio（反向偏离诊断）
对孤立拐角计算：
- `inside_ratio = mean(turn_sign*e_n ≥ 0)`
- `inside_depth = mean(clamp(turn_sign*e_n,0,ε/2))/(ε/2)`

判定：
- inside_ratio < 0.5：**先修符号/口径**（见 Step 1），禁止继续加内切奖励训练
- inside_ratio ≥ 0.6：可以进入 Step 2

---

## 3) Step 1：符号/口径对齐（消灭“反向偏离”）
> 目标：让 turn_sign 与 e_n 的“左正右负”口径一致，且在 corner_phase 段稳定不抖。

**要求：**
1) `e_n` 的定义固定：`e_n = dot(pos - proj, n_hat)`，`n_hat = [-t_y, t_x]`（左法向为正）
2) `turn_sign` 必须是局部几何意义的“左转/右转”  
   推荐从“进入拐角的入射切向 t_in 与出射切向 t_out”的 2D 叉积确定：  
   `turn_sign = sign( cross(t_in, t_out) )`（>0 左转，<0 右转）
3) corner_phase 只由 **dist_to_turn/turn_angle** 控制进入退出，不得由 e_n 或其他抖动量驱动

**验收（blocking）**：
- 在 deterministic test 的 corner_active/走廊 active 区间，`match_strict = mean(sign(e_n) == turn_sign)` ≥ 0.6  
  （允许少量 0 或噪声，门槛可在 B1 固定）

---

## 4) Step 2：让走廊 shaping 真正进入 reward（内切驱动力）
> 目标：在 corner_active 时，reward 中必须出现 “带内内切偏好 + 带外硬惩罚” 两类项。

在 `corridor_active=True`（corner_phase 或 exit ramp）时启用以下项：

1) **带外 barrier（硬惩罚）**  
- 依据 `dist_to_interval`（到走廊区间的距离）或 `|e_n| - ε/2`  
- 形式建议：`r_barrier = -w_barrier * (dist)^2`（可用 barrier_scale 调曲率）

2) **带内 center/target（软偏好）**  
- 依据 `e_target`（内侧半带的目标偏置，可取 0.6~0.9 倍半带深度）  
- 形式建议：`r_center = -w_center * |e_n - e_target|^p`（p=1 或 2）

3) **航向一致（lookahead heading）**  
- 使用 `heading_cos = cos(wrap(theta - theta_los))`  
- 形式建议：`r_heading = +w_heading * heading_cos` 或 `-w_heading*(1-heading_cos)`  
  只在 corridor_active 启用，避免直线段被扰动。

> 关键：这三项必须最终汇入 total reward，并落到 components/summary，避免“以为生效但没生效”。

**验收（必须）**：
- 在 `summary.json` 中能看到 corridor 相关分量非零（至少 barrier/heading/center 之一）
- inside_ratio 在训练后显著上升（或稳定 ≥ 0.6），且 max_abs_e_n 不变差

---

## 5) Step 3：让“圆弧/切线连续”成为可计算目标
新增（或固定）一组拐角形态指标，写入 `summary.json` 与 `main_table.csv`：

- `curvature_kappa_p95`：对 corner_mask 点列估计 κ≈Δθ/Δs，取 95 分位
- `corner_sharpness_index`：可直接等于 `curvature_kappa_p95` 或 `max(κ)`（越小越圆、越不尖）
- `inside_ratio / inside_depth`：见 Step 0
- `v_drop_ratio / min(v)`：拐角降速

**PASS（建议）**：
- inside_ratio ≥ 0.6
- corner_sharpness_index 相对 baseline 显著下降（你可在 B1 固定“显著”的量化口径）
- steps 不劣化（≤1.05× baseline）

---

## 6) Step 4：训练配方（避免“龟速圆弧”）
推荐做一个小的 curriculum（每次只动 1 个旋钮）：
1) 先开 barrier（防越带） + 轻量 heading（引导提前转向）
2) 再加 center/target（形成内切深度）
3) 最后才加 smoothness 强化（降低角加速度尖峰）

同时保留 B2a 的 deadzone：避免中心线惩罚把内切圆弧扼杀在摇篮里。

---

## 7) 交付物（和论文挂钩）
每个 run 必须产出：
- `overlay_corner_zoom.png`（baseline vs 当前 run）
- `summary.json`（含 inside_ratio / corner_sharpness_index / v_drop_ratio）
- `main_table.csv` 新增一行（可直接用于论文图表脚本）

---

## 8) Stop Rule（防止越训越歪）
- inside_ratio 连续 2 次 < 0.5：停止训练，回到 Step 1 修口径
- smoothness 变好但 v_drop 变差：说明策略在“慢速平滑”作弊 → 降低平滑权重或增加速度目标项
