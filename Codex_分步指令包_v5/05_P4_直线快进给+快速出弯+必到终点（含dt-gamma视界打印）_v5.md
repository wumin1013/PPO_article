# P4：直线快进给 + 快速出弯 + 必到终点（含 dt-gamma 视界检查）
> 前置条件：P3 corridor 已可运行且有可解释收益。  
> 本阶段实现“工程上可交付”的行为：直线段快、弯前降速、出弯快、且可靠到终点。

## 目标（Scope）
1) 直线快进给：turn-aware speed target  
2) 快速出弯：exit boost（有限步数窗口）  
3) 必到终点：抗停滞惩罚/截断 + 清晰 done 判据  
4) dt 与 gamma 耦合检查：打印有效视界（time horizon）

## 允许改动的文件
- `src/environment/cnc_env.py`（奖励、done、停滞检测、速度目标）
- （可选）`src/environment/reward.py`
- 新增：`tools/print_effective_horizon.py`（或在 main 启动时打印）

## 禁止改动（本阶段不要碰）
- P0/P1 的动作语义与一致性测试逻辑（不要回头改）
- 复杂几何内切参考线（仍然不要写）

## 任务 1：直线快进给（turn-aware speed target）
要求：速度目标 `speed_target` 随“到拐角距离/拐角角度”变化：
- 直线：`speed_target` 高（接近上限）
- 接近急弯：`speed_target` 平滑下降
- 出弯后 N 步：`speed_target` 快速回升（exit boost）

速度奖励建议：`-|v_ratio - speed_target|` 或平滑版本，替代固定目标速度比。

## 任务 2：快速出弯（exit boost）
- 检测“刚过弯”的事件（例如 segment_idx 跳变或 corner_phase 结束）；
- 在接下来 N 步增加“加速奖励/进度奖励权重”。

## 任务 3：必到终点（抗停滞 + done 判据）
- 强化终点接近奖励（progress 高时更陡）
- 停滞惩罚/截断：连续 K 步 progress_diff 很小（≈0）则惩罚或 truncated
- done 判据：progress 阈值 + 终点距离阈值（清晰可解释）

## 任务 4：dt 与 gamma 视界检查（必须；本检查已在 P0 前移，这里做回归确认）
实现 `tools/print_effective_horizon.py`：打印
- dt（interpolation_period）
- gamma
- time horizon 近似：例如 `H ≈ dt / (1 - gamma)`
并建议：若 dt 改变，按
- `gamma_new = gamma_old ** (dt_new / dt_old)`
保持每单位时间折扣一致（至少打印提醒，不一定自动改）。


## 默认参数与推荐公式（第一版起点，后续再调）
> 目标：给 Codex 一个“可直接编码”的默认实现，避免自由发挥到失控。

### 1) turn-aware speed_target（推荐一条可落地的标量函数）
用 lookahead 估计转向强度（与 P3 同口径）：
- `delta_theta = wrap(theta_far - theta_now)`
- `turn_severity = clip(abs(delta_theta) / theta_max, 0, 1)`，建议 `theta_max = 90°`

再结合“接近拐角程度”（若你已有 `dist_to_turn`，就用它；没有就用最近 lookahead 点的 `s_1` 近似）：
- `near = exp(-dist_to_turn / d_scale)`（dist_to_turn 越小，near 越接近 1）
- 建议 `d_scale` 取 lookahead 距离的 0.3~0.5 倍（同量纲）

速度目标（归一化到 [0,1] 的速度比）：
- `v_max = 1.0`（直线全速）
- `v_min = 0.35`（急弯最低目标，先保守）
- `speed_target = v_min + (v_max - v_min) * (1 - turn_severity * near)`

速度奖励建议（平滑、抗噪）：
- `r_speed = -w_speed * (v_ratio - speed_target)^2`（或 Huber），避免用尖锐的 L1 造成高方差。

### 2) exit boost（快速出弯窗口）
触发事件：`corner_phase` 从 True->False（或 segment_idx 跳变表示已过拐角）。  
窗口长度：`N_exit = max(5, int(0.25 / dt))`（大约 0.25s）。  
在窗口内可以做二选一：
- 提高 progress 奖励权重：`w_progress *= 1.2~1.5`
- 或额外给加速奖励：`+w_exit * max(0, v_ratio - v_prev_ratio)`

### 3) 抗停滞（必须可解释 + 可触发）
建议同时用“进度停滞”与“低速”双条件：
- `progress_diff < eps_p` 持续 K 步，且 `v_ratio < eps_v` → truncated / 重罚
- 建议：`eps_p = 1e-4`（按你 progress 定义再调），`eps_v = 0.05`
- `K = max(20, int(1.0 / dt))`（约 1s 的停滞才判死）

done 判据建议写清楚（避免“看起来到终点但没 done”）：
- `done_goal = (progress >= p_done) OR (dist_to_goal <= goal_tol)`
- 建议：`p_done = 0.995`，`goal_tol` 用你坐标系下的一个小阈值（例如路径单位的 1%~2%）。

### 4) dt-gamma 视界打印（你已经要求了，但这里补充打印项）
除 `H_time ≈ dt/(1-gamma)` 外，建议额外打印：
- `H_steps ≈ 1/(1-gamma)`
- 折扣半衰期：`half_life_steps = ln(0.5) / ln(gamma)`（直观）
并在 dt 改变时打印提醒：`gamma_new = gamma_old ** (dt_new / dt_old)`（保持“每单位时间折扣”一致）。

## 自验证/验收标准（你将这样验证）
### 验证 1：到终点率（硬指标）
- 用三条路径各训练/评估若干 episode：
- **验收：** 到终点率 >= 95%（max_steps 内）。

### 验证 2：速度模式符合预期（必须）
输出并检查曲线/统计：速度 vs 到拐角距离（或时间）
- **验收：** 直线段快、弯前慢、出弯快（趋势清晰）。

### 验证 3：停滞机制生效（必须）
构造“卡死”情形（低速度或原地震荡）：
- **验收：** 能触发停滞惩罚或截断，避免无意义拖回合。

### 验证 4：视界打印（必须）
运行任何一次训练启动：
- **验收：** 日志打印 dt、gamma、H 近似值，并在 dt 变化时给出提醒。

## 交付物（提交时必须包含）
1) 奖励项说明（speed_target、exit boost、停滞惩罚）  
2) done 判据说明  
3) 指标输出样例：到终点率、速度统计/曲线、停滞触发统计  
4) `tools/print_effective_horizon.py`
