# 05_P3.0 完成判据修复（open 终点、闭环 lap、进度单调）

> 插入位置：**P2.5 通过后、启用 P3 VirtualCorridor 之前**  
> 目的：把“能否到终点/跑完一圈”从玄学变成硬指标。  
> 你当前现象（高速常量动作也无法 success / lap 不触发 / 直线可能冲过终点仍不成功）通常不是训练问题，而是 **done/进度判定定义不稳定**。

---

## 0. 约束（必须遵守）
- **允许改动**：`src/environment/cnc_env.py`（仅限进度计算与终止判定相关函数）、`tools/` 下新增 sanity/debug 脚本。
- **禁止改动**：奖励、PPO 算法、网络结构、训练循环。
- Debug 打印默认关闭（或仅 episode 结束打印一次）。

---

## 1. 问题要修到什么程度（目标）
### 目标 A：Open path（非闭环）
- 直线/正方形开链/ S 形：**到达或越过终点**时，能够稳定 `success=True`。
- 禁止出现：“冲过终点 → end_distance 变大 → 永远不 success → timeout”。

### 目标 B：Closed path（闭环）
- 对 square（闭合）：**完成一圈**时稳定 `lap_completed=True` / `success=True`。
- 禁止出现：“速度再高也判不完圈”或“拐角附近最近段跳变导致 lap 永远不触发”。

---

## 2. 任务清单（按顺序做）

### Task 2.1 Open path：允许“越过终点线”触发成功（必须）
在进度计算/终止判定里实现任意一种（推荐 A）：

- **方案 A（推荐）**：在“最后一段”允许投影参数 `t` 不 clip 到 1。  
  - 当 `t >= 1.0` 且 `contour_error <= half_epsilon` → `reached_target=True` → done success  
  - `end_distance` 仅作诊断，不再作为强制 success 条件

- **方案 B**：保留 `t` clip，但在 `progress >= 0.999` 且 `contour_error <= half_epsilon` 时 success  
  - 注意：必须保证 `progress` 在越过终点后仍能到达 0.999，而不是卡死在某个值

> 关键：success 不应依赖“必须停在终点附近”，否则“快”永远学不会。

---

### Task 2.2 Closed path：改成“单调弧长累积 + 回到起点门控”（必须）
不要依赖“最近线段跳变覆盖率”这种不稳定判据。实现更鲁棒的闭环完成逻辑：

- 维护一个单调的 `s_travelled`（累计弧长/累计 progress），每 step 增量为沿当前段的前进量（投影差分）
- 当 `s_travelled >= (1 - tol)*total_length` 且 `dist_to_start < start_tol`，触发：
  - `lap_completed=True`
  - `success=True`

推荐默认阈值：
- `tol = 0.02`（允许 2%）
- `start_tol = 0.6 * half_epsilon`（与容差同尺度）

---

### Task 2.3 Episode 结束诊断字段（必须）
在 episode done 时（只打印一次）输出：
- `closed`, `lap_completed`, `reached_target`
- `final_progress`（或 `s_travelled/total_length`）
- `final_end_distance`（open）
- `final_dist_to_start`（closed）
- `final_contour_error`
- `mean_velocity`, `steps`, `dt`

---

### Task 2.4 常量动作 sanity（必须）
新增或更新 `tools/sanity_constant_action.py` 支持两种场景：
- `open_line` / `open_square`：应在 max_steps 内 success
- `square_closed`：应触发 lap_completed

参数示例：
- `--theta 0 --vel-ratio 0.8 --episodes 1 --seed 42`

---

## 3. 验收标准（必须全部满足）
1) Open path（line / open_square / s_shape）：常量动作 sanity 必须 `success=True`。  
2) Closed square：常量动作 sanity 必须 `lap_completed=True`（或 success=True）。  
3) 上述 sanity 全部在 **不训练**、仅用规则动作条件下通过。  
4) 通过后，才允许进入 P3（VirtualCorridor）与 P4（提速策略）。

---

## 4. 禁止事项
- 禁止为了让 success 触发而把 `half_epsilon` 或 `max_steps` 改得离谱。
- 禁止把 success 绑到“速度必须很小/必须停住”这类条件上。
