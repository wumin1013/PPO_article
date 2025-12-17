# 09_P5.1 真实几何 turn 估计：LOS 角（内切会影响的指标）+ corner/exit 事件对齐_v1
> 前置条件：P5.0 已通过（动作量纲统一，v_ratio_exec 正常）。  
> 本阶段目标：把 turn/kappa/phase 从“参考切向变化”升级为“**当前位置→预瞄点连线（LOS）几何**”，让**内切幅度变化会改变 turn 指标**，从而真正形成：  
> **走得更聪明（更内切/更顺）→ 转向需求更小 → 允许更快**。

---

## 目标（Scope）
1) 定义 LOS 角误差 `alpha`：由“当前位置→预瞄点”的连线角，与当前实际航向差得到  
2) 用 `alpha` 构造 kappa / turn_severity / corner_phase（带滞回）  
3) P4 的 exit boost 触发与 corner_phase 对齐：基于真实几何事件，而非参考路径事件  
4) 在 info 中输出 `alpha/L/kappa_los`，便于你检查“内切是否真的让 kappa 变小”

---

## 允许改动的文件
- `src/environment/cnc_env.py`
- （可选）`src/environment/reward.py`（若你把 turn 指标用于 reward 组件）
- 新增：`tools/p5_los_debug.py`（必做）

## 禁止改动（本阶段不要碰）
- corridor 奖励形状（P5.2 才改）
- 多预瞄 minimax + 角加速度/jerk 可控性边界（P6.0 才做）
- PPO 算法主体

---

## 任务 1：计算 LOS 角误差 alpha（必须）
### 1.1 选择预瞄目标点
使用你已有 lookahead 点（推荐第 K 个点，K=3~5）或用弧长推进取点：
- 得到 `P_far`（世界坐标）
- 计算 `r = P_far - current_position`
- `L = ||r||`（最小 eps）

### 1.2 LOS 角与航向
- `theta_los = atan2(r_y, r_x)`
- `heading = _current_direction_angle`（你在 apply_action 内已维护）
- `alpha = wrap(theta_los - heading)` ∈ [-pi, pi]

> 关键点：alpha 由当前位置决定；当轨迹内切，P_far 相对你的位置/朝向会改变，因此 alpha 会变。

---

## 任务 2：用 alpha/L 构造 kappa 与速度硬上限（必须）
### 2.1 kappa_los
- `kappa_los = |alpha| / (L + eps)`

### 2.2 turning-feasible cap（先用角速度约束版）
- `v_cap = MAX_ANG_VEL / (kappa_los + eps)`
- `v_ratio_cap = clip(v_cap / MAX_VEL, 0, 1)`

> 注意：本阶段只把“角速度约束”从参考切向改成 LOS；  
> “角加速度/角 jerk 可达性”留到 P6.0 再叠加。

---

## 任务 3：corner_phase 滞回改为基于 alpha（必须）
将 P3.1/P4 的 corner_phase 判定替换为：
- enter：`|alpha| > alpha_enter` 且 `L < L_enter`
- exit ：`|alpha| < alpha_exit` 或 `L > L_exit`
并满足：`alpha_exit < alpha_enter`，`L_exit > L_enter`（滞回）

建议初值：
- `alpha_enter = 15°`，`alpha_exit = 7°`
- `L_enter = 2~4 * lookahead_spacing`，`L_exit = 1.5 * L_enter`

---

## 任务 4：exit boost 触发对齐真实出弯事件（必须）
- 触发条件：`corner_phase` 从 True → False（由 alpha 滞回判定）
- 输出 debug：每当触发，打印一次：`step, alpha, L, v_ratio_exec, exit_window`

---

## 任务 5：新增 tools/p5_los_debug.py（必做）
脚本目的：证明 LOS 指标“对内切敏感”，而不是又变成参考切向的同义替代。

建议做两种对比（不需要训练）：
1) 用同一条路径（square），人为设置两种“侧向偏移”的起点（或短时间施加不同 theta_u）  
2) 记录拐角附近的序列：`e_n, alpha, kappa_los, v_ratio_cap`

验收要点（必须输出）：
- 在拐角段，`e_n` 变化时 `alpha/kappa_los` 也随之变化（不是常数/几乎不变）
- `v_ratio_exec <= v_ratio_cap` 恒成立

---

## 自验证/验收标准（你将这样验证）
1) **LOS 指标有效（硬指标）**  
   - 运行 `tools/p5_los_debug.py`  
   - **验收：** 拐角附近 `alpha/kappa_los/v_ratio_cap` 会随内切偏移变化而变化

2) **corner/exit 对齐（硬指标）**  
   - 记录 `corner_phase` 与 `alpha`：  
   - **验收：** 进入时 alpha 大，退出后 alpha 小且稳定；exit boost 只在退出时触发

3) **quick_eval 回归（必须）**  
   - line/square/s_shape 各 E=5  
   - **验收：** success 不显著下降；没有出现“莫名提前/延后出弯导致爆越界”的灾难性退化

---

## 交付物（提交时必须包含）
1) 改动文件列表 + 关键实现说明（alpha/L/kappa 的定义点）  
2) `tools/p5_los_debug.py` + 一段输出样例  
3) quick_eval 输出样例（line/square/s_shape）  
4) info 字段样例：`alpha, L, kappa_los, v_ratio_cap, corner_phase, exit_boost_remaining`
