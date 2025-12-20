# 人工复核清单：PPO_project P8.1（中间结果需要你看什么、怎么看）

> 这份文档是给你（人类）做“中间结果复核/放行训练”的。  
> Codex 可以做自动化断言，但**有些目标（平滑、最小降速、内切是否合理）需要你肉眼和曲线判断**。

---

## 1) 哪些可以完全交给 agent 自己验证（无需你看）
这些由脚本强断言即可：
- state 语义绑定一致：state[3]=dist_to_turn、state[5]=turn_angle  
- dist_to_turn 是 Arc Length（且与欧式距离明显不同）  
- corner_phase 在 corridor 关闭时不被污染  
- reset 初始化正确（无开局脉冲）  
- lookahead 不饱和、无 NaN/Inf、动作/状态范围裁剪正确

对应文件：
- `tools/accept_p8_1_observation_and_corner_phase.py` 的断言与 summary.json

---

## 2) 哪些必须人工复核（Phase A 结束的放行门禁）
### 2.1 Step 3：专家策略闭环（训练前放行门禁）
你要看两张图（Square 路径）：

1) `artifacts/square_e_n.png`
- 合格：e_n(t) 基本落在 ±half_epsilon 内，最多短暂贴边，不应长时间越界
- 红旗：
  - e_n 持续单向漂移（说明可控性/heading/切线角初始化仍有问题）
  - e_n 在拐角处出现高频振荡（说明角速度/平滑项或观测前瞻仍不稳定）

2) `artifacts/square_v_ratio_exec.png`
- 合格：直线段 v_ratio_exec 明显升高；接近拐角前按 braking/cap 合理下降；出弯后能再升高
- 红旗：
  - v_ratio_exec 在直线段长期很低（说明 v_cap/speed_target 或奖励门槛仍把速度压死）
  - 拐角前不降速然后出界（说明 dist_to_turn/turn_angle 或 braking 包络失效）
  - 出弯后速度起不来（可能 exit_boost/corner_phase 或速度奖励 gating 有问题）

同时看 `summary.json`：
- 到终点：True
- max_abs_e_n：最好 <= half_epsilon
- stall：False

> 这一步通过，才允许 agent 进入 Step 5 正式训练。

### 2.2 轨迹图（path.png）
- 合格：拐角处是圆滑过渡，不是尖角折线；出弯不贴边乱抖
- 红旗：明显尖角、或“拐不过来”外抛、或 S 型中线穿墙

---

## 3) 关于“最大效率（最小降速）”你该怎么看
训练后你要看的不是“reward 变大没”，而是这些可解释指标：
- mean_speed_util：在带内的平均 v_exec/(v_cap+eps) 越接近 1 越好
- progress_per_step：平均每步弧长进度越大越好
- e_n 的分布：不能为了速度长期贴边或越界

> 目标不是“永远满速”，而是“物理可行下尽量少降速、且过弯平滑”。

---

## 4) 放行规则（给你一个不纠结的标准）
- Step 3（专家策略）在 Square 上：
  - e_n 曲线稳定（无漂移、无大幅越界、无开局尖峰）
  - v_ratio_exec 符合“直线升、弯道降、出弯再升”
  - summary 显示到终点、无 stall
=> 允许进入 Step 5 训练。

否则：先让 agent 回到 Runbook 排查（优先动力学模式、reset 初始化、Arc dist、corner_phase 隔离、归一化链路）。


> 放行提示：只有当你在 Phase A 的专家策略验证中认可了 Square 的 e_n 与 v_ratio_exec 曲线，才允许进入 Phase B（tangent_relative/效率奖励/正式训练）。
