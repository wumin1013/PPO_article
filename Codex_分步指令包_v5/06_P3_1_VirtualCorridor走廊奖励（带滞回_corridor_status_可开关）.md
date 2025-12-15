# 06_P3.1 VirtualCorridor 走廊奖励（带滞回、corridor_status、可开关）
> 前置条件：P2 已通过（lookahead 特征稳定、训练不崩）。  
> 本阶段只实现 **鲁棒 Fallback**：内切走廊奖励。**不要上复杂几何偏移线/参考弧**（风险太高）。

## 目标（Scope）
- 在拐角阶段允许轨迹在“中心线与内侧边界”之间内切，通过 reward 引导在走廊内寻找误差-效率折中；
- 走廊逻辑必须**可开关**，便于消融与回退。


## [新增强制要求] corner_phase 滞回 + corridor_status 可观测（必做）

为避免“第一个拐角冲出去/走廊符号搞反却看不出来”的问题，本阶段额外强制：

1) **corner_phase 必须带滞回（enter/exit 两阈值）**  
   - enter: `abs(next_angle) > angle_enter` 且 `dist_to_turn < dist_enter`  
   - exit : `abs(next_angle) < angle_exit` 或 `dist_to_turn > dist_exit`（`exit < enter` 形成滞回）  
   - 默认建议：`angle_enter=20°`, `angle_exit=10°`, `dist_enter=2~4*lookahead_step`, `dist_exit=dist_enter*1.5`

2) **info 必须返回 corridor_status**（用于自检/日志样例）  
   - `turn_sign, e_n, lower, upper, e_target, in_corridor(bool)`  
   - 且提供日志样例证明：左转时 `lower>=0`、右转时 `upper<=0`（符号自洽），并能解释 `e_n` 的正负含义。

> 走廊是“看不见”的概念。没有 corridor_status，就等于蒙眼调参。


## 允许改动的文件
- `src/environment/cnc_env.py`（corner_phase 判定、走廊误差、奖励项）
- （可选）`src/environment/reward.py`（若你把奖励模块化）

## 禁止改动（本阶段不要碰）
- 几何偏移线/内缩多边形/圆弧过渡算法（不要写）
- P4 的速度目标与停滞终止逻辑（留到下一阶段）

## 任务：Virtual Corridor 走廊奖励（必须）
### 核心逻辑
1) 计算转向方向 `turn_sign`（左转 +1，右转 -1）。  
2) 计算相对中心线的有符号法向误差 `e_n`（中心线=0，左正右负）。  
3) 定义走廊（含 margin）：  
   - 左转：`e_n ∈ [0, half_epsilon - margin]`  
   - 右转：`e_n ∈ [-half_epsilon + margin, 0]`  
4) 奖励：
   - 在走廊内：tracking 给高分（不再强迫贴中心线）；
   - 走廊外：按越界距离连续惩罚（必要时终止）；
   - 加朝向一致性：`cos(angle(v_dir, tangent_future))`；
   - 叠加进度奖励（progress 增长才给分）。


### corner_phase 判定（建议默认实现：带滞回，避免抖动）
> 目标：只在“确实接近拐角且转向明显”的阶段开启 corridor；拐角结束后迅速关闭。

推荐用 lookahead 的“未来切向变化”作为 turn 强度：
- `theta_now = path_tangent_angle(s_now)`
- `theta_far = path_tangent_angle(s_now + s_lookahead)`（用第 K 个 lookahead 点的弧长/位置估计）
- `delta_theta = wrap(theta_far - theta_now)`（范围 [-pi, pi]）
- `turn_sign = sign(delta_theta)`（左转 +1，右转 -1；接近 0 时不进入拐角）

进入/退出阈值（给一套初始值，后续可调）：
- 进入：`abs(delta_theta) > theta_enter` 且 `dist_to_turn < d_enter`
- 退出：`abs(delta_theta) < theta_exit` 或 `dist_to_turn > d_exit`
- 建议初始：`theta_enter = 15°`，`theta_exit = 8°`；`d_enter/d_exit` 用“路径归一化距离”或“lookahead 物理距离”的 0.2~0.4 倍（两者保持同量纲）。

### 参数默认值（建议作为第一版起点）
- `margin = 0.1 * half_epsilon`（并 clamp 到 [0, 0.4*half_epsilon]）
- 走廊“目标偏置”（可选但常有效）：  
  `e_target = 0.5 * (half_epsilon - margin) * turn_sign`（引导在走廊中间偏内侧，而不是贴边）
- 走廊内奖励建议用平滑项（避免硬阈值导致不可导/高方差）：  
  - 在走廊内：对 `|e_n - e_target|` 轻惩罚（而不是对 `|e_n|`）  
  - 在走廊外：按“到区间的距离”二次惩罚 `dist_to_interval^2`（连续可微）

### 配置开关（必须）
- 增加配置项，例如：`experiment.enable_corridor` 或 `reward.corridor.enabled`；
- 关闭时应退回旧版 tracking 逻辑（中心线误差）。

## 自验证/验收标准（你将这样验证）
### 验证 1：开关有效（必须）
- 开 corridor 与关 corridor，各跑 50 episode：
- **验收：** 两种模式都能跑；开 corridor 时拐角处轨迹出现内侧偏置趋势。

### 验证 2：越界率不恶化（必须）
- 统计越界次数/越界终止比例：
- **验收：** corridor 开启后越界率不显著上升（或下降）。

### 验证 3：效率/到终点率改善（建议）
- 统计到终点率、平均用时/步数、平均速度：
- **验收：** corridor 开启后到终点率或效率明显改善。

### 可视化/可观测验证（必须补充，防止符号搞反）
在 `step()` 的 `info` 中额外返回 `corridor_status`（用于日志/plot）：
- `enabled`：是否启用 corridor（bool）
- `turn_sign`：左转 +1 / 右转 -1
- `e_n`：当前有符号法向误差（左正右负）
- `lower` / `upper`：当前 corridor 边界（同 e_n 的符号约定）
- `e_target`：目标内切偏置（例如左转取正，右转取负）

**验收补充：**
- 给出一段日志样例（或 plot 截图）证明：
  - 左转时 corridor 生成在中心线左侧：`turn_sign=+1` 且 `lower>=0`、`upper>0`
  - 右转时 corridor 生成在中心线右侧：`turn_sign=-1` 且 `upper<=0`、`lower<0`
- 若 Agent 退化成“永远贴中心线”，也要能从 `corridor_status` 看出 corridor 是否真的开启、边界是否合理。
## 交付物（提交时必须包含）
1) corridor 公式与参数说明（margin、corner_phase 判定等）  
2) 开关配置说明 + 示例配置  
3) 指标输出：越界率、到终点率、平均速度/步数（至少打印）