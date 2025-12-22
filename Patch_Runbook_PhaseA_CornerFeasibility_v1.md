# Phase A 补丁指令 #2（进入 Phase B 前必须完成）
## 主题：拐角可行性（Corner Feasibility）——把“能跑得快”升级为“快了也转得过去、不越界”

> 背景：在补丁 #1 之后，专家策略已经能**跟随 v_ratio_cap**，速度维度不再退化为常数龟速。  
> 但这会暴露一个更真实的瓶颈：**拐角/急弯处的转向可行性**。  
> 如果 v_ratio_cap 只考虑“刹车距离”而没有考虑“角速度上限/可实现曲率”，就会出现：  
> **速度利用率很高（speed_util≈1）但拐角仍然顶到允差带边界甚至越界**，Square/S 训练会很难收敛。

本补丁的核心：在 v_ratio_cap 中加入**角速度可行性上限**（Angular Feasibility Cap），并让专家策略在拐角处采用一个“最小模型”的可行过弯控制律，以保证 Square 专家验证稳定到终点。

---

# 0) 本补丁的 Definition of Done（DoD）
必须全部满足：

## 0.1 自动化验收全绿（回归）
```bash
cd PPO_project
python tools/check_physics_logic.py
python tools/accept_p7_0_dynamics_and_scale.py
python tools/accept_p8_1_observation_and_corner_phase.py
python -m pytest -q
```

## 0.2 Square 专家策略必须“到终点且不越界”
在 `tools/accept_p8_1_observation_and_corner_phase.py` 的 Square 验证中：
- `reached_target == True`
- `stall_triggered == False`
- `max_abs_e_n <= 0.98 * half_epsilon`（留 2% 安全裕度，避免擦边算通过）

## 0.3 速度效率不能被“过度保守”掐死
Square 专家策略仍需满足：
- `mean_speed_util_in_band >= 0.70`  
（若你的单位/滤波导致略低，可把阈值调到 0.65，但必须留门禁）

## 0.4 产物齐全（用于人工复核）
在 `artifacts/phaseA/` 下必须包含：
- `path.png`
- `square_e_n.png`
- `square_v_ratio_cap.png`
- `square_v_ratio_exec.png`
- `square_speed_util.png`
- **新增（必须）**：`square_v_cap_breakdown.png`（见 3.2）
- `summary.json` 与 `trace.csv`

---

# 1) 需要修改的代码位置（用 grep 定位，禁止“拍脑袋改错文件”）
你必须先定位 v_ratio_cap 的来源。请在 repo 根目录执行：

```bash
rg -n "v_ratio_cap" -S
rg -n "braking" -S
rg -n "dist_to_turn" -S
rg -n "turn_angle" -S
```

常见落点（以你的项目为准）：
- `src/environment/cnc_env.py`（step/info/status 计算）
- `src/environment/reward.py`（如果 cap 参与奖励或 gating）
- `src/environment/*` 中的 `_compute_p4_pre_step_status` / `braking_envelope` / `scan_for_next_turn` 类函数

> 要求：Angular Feasibility Cap 必须写在**生成 v_ratio_cap 的同一处**，确保 PPO、专家策略、trace、dashboard 看到的是同一个 cap。

---

# 2) 补丁核心：Angular Feasibility Cap（必须）
## 2.1 设计目标
对任意时刻给定：
- `turn_angle`：未来拐角的方向变化（弧度，带符号）
- `epsilon`：走廊总宽度（你的设置里 half_epsilon = epsilon/2）
- `MAX_ANG_VEL`：角速度上限
- `MAX_VEL`：线速度上限

给出一个“保证可转得过”的速度上限 `v_cap_ang`，并与现有 `v_cap_brake` 合并：
- `v_cap_final = min(v_cap_brake, v_cap_ang)`
- `v_ratio_cap_final = v_cap_final / MAX_VEL`

## 2.2 推荐实现（稳健且不依赖 dist_to_turn 的分母爆炸）
**核心思想**：拐角可实现的“最小转弯半径”由走廊宽度决定。  
对一个角度变化 `|turn_angle|`，如果你允许在走廊内做圆弧过渡，可以用一个“可行半径上界”近似：

- `half_eps = 0.5 * epsilon`
- `sin_half = sin(min(abs(turn_angle)/2, pi/2))`  # 防止极端角度数值问题
- `r_allow = half_eps / max(sin_half, sin_min)`  
  - 建议 `sin_min = 0.2`（防止很小的 turn_angle 使 r_allow 过大导致 cap 失效；小弯本来也不需要太保守）
- 角速度约束：`omega = v / r`（圆弧上）
- 所以 `v_cap_ang = MAX_ANG_VEL * r_allow`

然后：
- `v_cap_final = min(v_cap_brake, v_cap_ang)`
- `v_ratio_cap = v_cap_final / MAX_VEL`

> 直觉校验：  
> - 对 Square（turn_angle≈pi/2），`sin(pi/4)=0.707`，`r_allow≈1.41*half_eps`，比直接用 half_eps 稍不保守。  
> - 对更急的拐角，r_allow 不会变得离谱；对小角度缓弯，r_allow 会变大，cap 不会无意义地压低。

### 强制护栏（必须）
- `v_cap_ang` 必须有下限：`v_cap_ang >= v_min`（建议 `v_min_ratio=0.01` → `v_min=v_min_ratio*MAX_VEL`），防止数值上“完全停住”导致 stall。
- 所有计算必须避免 NaN/Inf（对 turn_angle=0 直接返回 `v_cap_ang=+inf` 或 `MAX_VEL`）。

---

# 3) 专家策略过弯的“最小可行控制律”（必须）
> 目的：Phase A 的专家验证必须稳定到终点，给 Phase B 一个可靠 baseline。  
> 注意：这不是为了“专家最优”，而是为了“专家可行且不越界”。

## 3.1 在接近拐角时，专家策略应进入 Corner Mode
请在专家策略中（通常在 `tools/accept_p8_1_observation_and_corner_phase.py` 或 expert 模块）加入：

- 计算 fillet 的“提前入弯距离”（几何上圆弧与直线相切的距离）：
  - `half_eps = epsilon/2`
  - `sin_half = sin(abs(turn_angle)/2)`
  - `r_allow = half_eps / max(sin_half, sin_min)`
  - `d_fillet = r_allow * tan(abs(turn_angle)/2)`  # 需要开始转向的提前距离（沿弧长近似）

- 当 `dist_to_turn <= d_fillet` 时进入 Corner Mode：
  - 速度命令仍按补丁 #1：`v_ratio_cmd = clip(0.9*v_ratio_cap, v_ratio_min, 1.0)`
  - 转向命令采用圆弧可行解：
    - `omega_target = sign(turn_angle) * min(MAX_ANG_VEL, v_exec / max(r_allow, r_min))`
    - `omega_ratio_cmd = omega_target / MAX_ANG_VEL`

> 直觉：corner mode 下你是在“尽可能用一个可行半径 r_allow 的圆弧”去拐，而不是临近拐点才用 PD 抢救。  
> 这会显著减少“顶到半带边界”的风险。

## 3.2 新增可视化：v_cap 分解图（必须）
在 Square 验证脚本结束时，除了原有图外，新增：
- `square_v_cap_breakdown.png`：画三条曲线
  - `v_ratio_cap_brake(t)`
  - `v_ratio_cap_ang(t)`
  - `v_ratio_cap_final(t)`（实际用于 env 的 cap）

> 这张图是定位神器：  
> - 如果角速度 cap 从不生效（final==brake），说明 ang cap 太松或 turn_angle 没接上；  
> - 如果角速度 cap 一直压死（final==ang 且很低），说明 r_allow 设计过保守或 sin_min 太大。

---

# 4) trace.csv / summary.json 必须新增字段（便于自动验收与定位）
## 4.1 trace.csv 新增列（列名固定）
必须新增（若已存在可复用，但必须写入）：
- `v_ratio_cap_brake`
- `v_ratio_cap_ang`
- `v_ratio_cap`（final）
- `corner_mode`（0/1，专家策略是否进入 corner mode）
并保持已有：
- `dist_to_turn, turn_angle, v_ratio_exec, e_n, omega_exec`

## 4.2 summary.json 新增指标（最少这些）
- `reached_target`
- `stall_triggered`
- `max_abs_e_n`
- `mean_speed_util_in_band`
- `corr_v_exec_vs_v_cap`
- `cap_ang_active_ratio`：final 由 ang cap 主导的 steps 比例（例如 `mean(v_ratio_cap_ang < v_ratio_cap_brake)`）

---

# 5) 新增强门禁断言（必须）
在 `tools/accept_p8_1_observation_and_corner_phase.py` 的 Square 验证中加入断言：

- A：必须到终点  
  - `assert reached_target`

- B：必须带内（留 margin）  
  - `assert max_abs_e_n <= 0.98*half_epsilon`

- C：速度效率不能太差  
  - `assert mean_speed_util_in_band >= 0.70`

- D：角速度 cap 必须“有机会生效”（避免实现了但从不触发）  
  - `assert cap_ang_active_ratio >= 0.05`  
  （若你的 braking envelope 本来就足够保守导致 ang cap 很少触发，可把阈值调低，但不能为 0）

所有断言失败必须输出定位提示：
- turn_angle 是否为 0
- v_ratio_cap_brake/ang/final 的统计（min/mean/max）
- corner_mode 进入比例

---

# 6) 是否需要人为验收？
需要，但非常轻量（看 3 张图即可）：

1) `square_v_cap_breakdown.png`：final 是否在拐角前被 ang cap 适度压低（不是全程压死）  
2) `path.png`：拐角是否圆滑且不擦边越界  
3) `square_e_n.png`：是否留有 margin（不要经常贴着 ±half_epsilon）

人类验收通过后，才允许进入 Phase B。

---

# 7) 常见失败模式与修正建议（给 agent 自验证）
- **仍然越界**：  
  - 降低 `sin_min`（让 r_allow 更小 → 更保守）或提高 corner mode 触发提前量（增大 d_fillet）  
  - 确认 omega 在 corner mode 下没有被其他模块覆盖（检查 omega_exec 是否达到 omega_target）

- **stall/推进过慢**：  
  - 提高 `v_min_ratio`（例如 0.02）  
  - 或减小过保守的 ang cap（调大 sin_min 或给 r_allow 下限）

- **ang cap 从不生效**：  
  - 说明 turn_angle/dist_to_turn wiring 可能不对，或 braking envelope 已经远保守过头  
  - 先检查 `v_ratio_cap_brake` 是否长期很低

---

# 8) 强制执行语句（必须原样保留）
请严格执行这份指令。在修改任何代码之前，先确保你理解了行为目标：  
**直线尽可能高速、拐角前基于允差与角速度上限自动降速、通过可行圆弧平滑过弯、并在允差带内到达终点。**  
在提交任何修改后，必须运行 `tools/accept_p8_1_observation_and_corner_phase.py` 并提供 `square_v_cap_breakdown.png`、`path.png`、`summary.json` 证明拐角可行性与速度效率同时成立。失败时优先检查：turn_angle/dist_to_turn 的物理含义是否对齐、以及角速度可行性 cap 是否正确生效。
