# 07_P4.0 直线快进给 + 快速出弯 + 必到终点（含 dt-gamma 视界打印）_v6
> 前置条件：P3 corridor 可运行且 done 判据已修复（支持 open-path 过冲成功）。  
> 目标：工程上可交付的行为：**直线段快、弯前降速、出弯快、且可靠到终点**。  
> 新增硬约束：**角速度约束 → 速度上限**，让“弯前降速”不只是奖励引导，而是物理可转向性约束。

---

## 目标（Scope）
1) 直线快进给：turn-aware speed target  
2) 快速出弯：exit boost（有限步数窗口）  
3) 必到终点：抗停滞惩罚/截断 + 清晰 done 判据  
4) dt 与 gamma 耦合检查：打印有效视界（time horizon）  
5) [新增] Turning-feasible speed cap：由 `MAX_ANG_VEL` 推出 `v_cap`

---

## [强制要求] success 优先级最高（先到终点，再谈更快）
本阶段所有“提速”策略都必须以 **success_rate 不下降** 为硬约束。为避免训练学成“龟速保命”或“过冲不成功”，强制：

1) **时间惩罚（time_penalty）**：每步固定扣分（默认 -0.01），逼迫更快完成；  
2) **停滞终止（stall termination，可开关）**：连续 N 步 progress 增量 < tiny 直接 done 并给负奖励；  
   - 默认：`N=300`, `tiny=1e-4`  
3) **过冲成功兼容**：若 P3.0 已实现“越过终点线可成功”，这里不得再把 success 绑到 end_distance 很小。

---

## 允许改动的文件
- `src/environment/cnc_env.py`（奖励、done、停滞检测、速度目标、速度硬上限）
- （可选）`src/environment/reward.py`
- 新增：`tools/print_effective_horizon.py`（或在 main 启动时打印）

## 禁止改动（本阶段不要碰）
- P0/P1 的动作语义与一致性测试逻辑（不要回头改）
- 复杂几何内切参考线（仍然不要写）

---

## 任务 1：直线快进给（turn-aware speed target）
要求：速度目标 `speed_target` 随“到拐角距离/拐角角度”变化：
- 直线：`speed_target` 高（接近上限）
- 接近急弯：`speed_target` 平滑下降
- 出弯后 N 步：`speed_target` 快速回升（exit boost）

速度奖励建议：`r_speed = -w_speed * (v_ratio_exec - speed_target)^2`（或 Huber）。

> 注意：本版本引入 `v_ratio_exec`（执行速度比），用于体现硬上限裁剪后的真实速度。

---

## 任务 1.5（新增）：角速度约束 → 速度上限（Turning-feasible speed cap）
目的：防止策略学出“想转但转不过去”的高风险快法。

### 1.5.1 用 lookahead 估计未来曲率
- 取 lookahead 弧长：`s_lookahead`（建议 = lookahead_points 的累计弧长，或 spacing*k，k=3~5）
- 取远点期望航向：`theta_far`（来自 lookahead 点的切向/连线角）
- 当前航向：`theta_now`
- `delta_theta = wrap(theta_far - theta_now)`
- 曲率估计：`kappa = abs(delta_theta) / (s_lookahead + eps)`

### 1.5.2 由最大角速度推出速度上限
- `v_cap = MAX_ANG_VEL / (kappa + eps)`
- 归一化速度比上限：`v_ratio_cap = clip(v_cap / MAX_VEL, 0, 1)`

### 1.5.3 执行侧裁剪（必须）
若策略输出为 `v_ratio_policy`（速度比动作）：
- `v_ratio_exec = min(v_ratio_policy, v_ratio_cap)`
- 速度奖励/动力学/动作执行都使用 `v_ratio_exec`（不是 policy 原值）
- 仍可在 info 中记录 `v_ratio_policy` 便于诊断

若速度不是动作而是由环境内部生成：
- `speed_target = min(speed_target, v_ratio_cap)`（硬上限压住目标）

### 1.5.4 Debug 打印（默认关闭，可开关）
在 `info` 或 debug 日志里输出：
- `kappa, v_cap, v_ratio_cap, v_ratio_policy, v_ratio_exec, dist_to_turn, turn_severity`

> 预期：square 第一拐角越界率明显下降；同时直线段仍可接近满速。

---

## 任务 2：快速出弯（exit boost）
- 检测“刚过弯”的事件（例如 `corner_phase` 从 True->False，或 segment_idx 跳变）
- 在接下来 N 步提高“加速/进度”收益（窗口短、力度有限，避免震荡）

---

## 任务 3：必到终点（抗停滞 + done 判据）
- 强化终点接近奖励（progress 高时更陡）
- 停滞惩罚/截断：连续 K 步 progress_diff 很小（≈0）则惩罚或 truncated
- done 判据：**以 progress/终点穿越为主**，不要再强绑 end_distance 很小

---

## 任务 4：dt 与 gamma 视界检查（必须）
实现 `tools/print_effective_horizon.py`：打印
- dt（interpolation_period）
- gamma
- 近似有效视界（time horizon）：`H_time ≈ dt / (1 - gamma)`
并提示：若 dt 改变，按
- `gamma_new = gamma_old ** (dt_new / dt_old)`
保持每单位时间折扣一致（至少打印提醒）

---

## 默认参数与推荐公式（第一版起点）
### 1) turn-aware speed_target
- `delta_theta = wrap(theta_far - theta_now)`
- `turn_severity = clip(abs(delta_theta) / theta_max, 0, 1)`，建议 `theta_max = 90°`
- `near = exp(-dist_to_turn / d_scale)`（dist_to_turn 越小 near 越接近 1）
- `v_max = 1.0`，`v_min = 0.35`
- `speed_target = v_min + (v_max - v_min) * (1 - turn_severity * near)`

然后应用硬上限：
- `speed_target = min(speed_target, v_ratio_cap)`（或裁剪 `v_ratio_exec`）

### 2) exit boost
- 触发：`corner_phase` True->False
- `N_exit = max(5, int(0.25 / dt))`
- 窗口内：提高 progress 权重（如 `w_progress *= 1.2~1.5`）或奖励加速趋势

### 3) 抗停滞
- 条件：`progress_diff < eps_p` 持续 K 步，且 `v_ratio_exec < eps_v`
- 建议：`eps_p = 1e-4`，`eps_v = 0.05`
- `K = max(20, int(1.0 / dt))`（约 1s）

### 4) done 判据建议（避免“到终点但没 done”）
- `done_goal = (progress >= p_done) OR crossed_goal_line`
- `p_done = 0.995`（按你的 progress 定义微调）
- open-path 若支持过冲成功：`crossed_goal_line` 必须可触发 success

---

## 自验证/验收标准（你将这样验证）
### 验证 1：到终点率（硬指标）
- line / square / S 各训练+评估若干 episode  
- **验收：** 到终点率 >= 95%（max_steps 内）

### 验证 2：速度模式符合预期
输出统计/曲线：速度 vs 到拐角距离（或时间）
- **验收：** 直线段快、弯前慢、出弯快（趋势清晰）

### 验证 3：速度硬上限确实生效（新增必测）
在 square 急弯附近打印/记录：
- `kappa` 升高时 `v_ratio_cap` 下降，且 `v_ratio_exec <= v_ratio_cap` 恒成立  
- **验收：** 急弯处的实际速度被压住，越界率下降

### 验证 4：停滞机制生效
构造“卡死”情形（低速/原地震荡）：
- **验收：** 触发停滞惩罚或截断，避免无意义拖回合

### 验证 5：视界打印
启动任意训练：
- **验收：** 打印 dt、gamma、H_time，并在 dt 变化时给出提醒

---

## 交付物（提交时必须包含）
1) 奖励项说明（speed_target、exit boost、停滞惩罚、速度硬上限）  
2) done 判据说明（含 open-path 过冲成功兼容）  
3) 指标输出样例：到终点率、速度统计/曲线、停滞触发统计、硬上限裁剪统计  
4) `tools/print_effective_horizon.py`
