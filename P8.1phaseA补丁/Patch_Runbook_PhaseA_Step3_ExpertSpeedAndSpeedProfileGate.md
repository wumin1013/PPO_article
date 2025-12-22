# Phase A 补丁指令（必须在进入 Phase B 前完成）
## 主题：修复专家策略的速度闭环 + 增加“速度剖面门禁”验收（避免 PhaseA 虚通过）

> 你当前 Phase A 专家策略验证出现了一个典型“虚通过”模式：  
> **v_ratio_exec 全程几乎恒定且极低**，导致验证只证明“低速不炸”，并没有验证你的 **Braking Envelope / v_cap / speed_target** 能驱动“直线快、弯前减速、出弯再提速”的速度剖面。  
> 本补丁要求：**让专家策略主动贴近 v_ratio_cap（或 speed_target_ratio）跑**，并用脚本门禁强制验证速度剖面真的存在。

---

# 0) Definition of Done（补丁 DoD）

必须全部满足：

### 0.1 自动化验收全绿
```bash
cd PPO_project
python tools/check_physics_logic.py
python tools/accept_p7_0_dynamics_and_scale.py
python tools/accept_p8_1_observation_and_corner_phase.py
python -m pytest -q
```

### 0.2 必须产出（默认 Square 配置）
在 `artifacts/phaseA/` 目录下必须新增/更新：
- `square_v_ratio_exec.png`（已有）
- `square_e_n.png`（已有）
- **新增**：`square_v_ratio_cap.png`（上限速度剖面）
- **新增**：`square_speed_util.png`（速度利用率 = v_exec/(v_cap+eps)）
- `trace.csv`（必须包含新增字段，见 2.2）
- `summary.json`（必须包含新增指标，见 2.3）

### 0.3 新的“速度剖面门禁”必须通过（关键）
**默认 Square** 上，专家策略必须满足（建议阈值，若单位不同可微调，但必须有门禁）：
- `v_ratio_exec_nunique >= 10`（速度不是常数，至少有结构性变化）
- 直线段平均 `speed_util` ≥ 0.5（说明会“用速度”而不是龟爬）
- 拐角前减速确实发生：存在一段 steps 使 `v_ratio_cap` 相对直线段明显下降，且 `v_ratio_exec` 随之下降（相关性要为正，见 3.2）

> 注意：这些门禁不要求“最优”，只要求“速度维度活了”。

---

# 1) 需要修改的文件与位置（强制）

你必须在 **专家策略**与**验收脚本**两个地方同时改，否则等于没修。

## 1.1 专家策略位置
在你项目中，专家策略通常位于：
- `tools/accept_p8_1_observation_and_corner_phase.py` 内部  
或  
- `src/` 下某个 `expert_policy.py` / `controllers/` 模块

你需要找到代码里类似下面的逻辑（现状往往是固定速度）：
- `v_ratio = constant` 或 `v_action = constant`
- `omega` 基于 `tau_next` 或 `e_n` 做 PD

---

# 2) 补丁内容（必须按顺序做）

## 2.1 让专家策略速度贴近 v_ratio_cap（或 speed_target_ratio）
### 2.1.1 推荐实现（最简单、最符合你的目标）
在每个 step，先获得当前的 P4/P8 状态（必须来自环境的同一套逻辑）：

- 方式 A（推荐）：调用环境已有函数  
  - `status = env._compute_p4_pre_step_status()`  
  - 取 `v_ratio_cap = status["v_ratio_cap"]` 或 `speed_target_ratio = status["speed_target_ratio"]`

- 方式 B（如果 A 不可访问）：从 `info` 里读（需要 env.step() 把这些塞进 info）  
  - `info["p4"]["v_ratio_cap"]`、`info["p4"]["speed_target_ratio"]`

然后定义专家速度命令（核心）：
- `v_ratio_cmd = clip(k_cap * v_ratio_cap, v_ratio_min, 1.0)`

建议参数：
- `k_cap = 0.9`（留一点裕度，避免频繁顶到裁剪）
- `v_ratio_min = 0.02`（防止数值上“停住”导致 stall 或无法推进）
- 可选平滑：`v_ratio_cmd = 0.8*v_ratio_cmd_prev + 0.2*v_ratio_cmd`

> 若你的系统有加速度限制/速度滤波，v_ratio_exec 可能不等于 v_ratio_cmd，这是正常的；但 v_ratio_exec 必须呈现“直线高、弯前低”的结构。

### 2.1.2 转向策略保持简单即可（别引入新规则）
保持现有 PD（例如基于 tau_next）即可：
- `omega_ratio_cmd = clip(-k_tau * tau_next / MAX_ANG_VEL, -1, 1)`
- 不要在这个补丁里改转向逻辑（避免变量过多）

---

## 2.2 trace.csv 必须新增字段（用于自动验收定位）
在 trace 记录中必须新增列（列名固定，便于脚本断言）：
- `v_ratio_cmd`（专家输出）
- `v_ratio_cap`（环境计算的上限/目标约束）
- `speed_util`（= v_ratio_exec / (v_ratio_cap + 1e-9)）
- `dist_to_turn`、`turn_angle`（若已存在可复用；没有必须加）
- `corner_phase`（True/False，若可获得）

并确保 trace 中仍包含：
- `v_ratio_exec`、`e_n`

> 这能在失败时快速定位：是 expert 没跟 cap，还是 cap 本身不变，还是执行层滤波过强。

---

## 2.3 summary.json 必须新增指标（用于门禁）
在 summary 里新增并输出（至少这些）：
- `v_ratio_exec_nunique`
- `mean_speed_util_in_band`（只统计 |e_n|<=half_epsilon 的 steps）
- `mean_v_ratio_exec_straight` 与 `mean_v_ratio_exec_near_turn`  
  - 直线段可用条件：`dist_to_turn > dist_straight_threshold`（例如 2.0 或按尺度）
  - 近拐角段：`dist_to_turn < dist_turn_threshold`（例如 0.5 或按尺度）
- `corr_v_exec_vs_v_cap`（Pearson 相关系数，期望为正，且不要接近 0）

---

# 3) 自动化验收（必须写进 accept 脚本）

## 3.1 新增图表输出（默认 Square）
在 `tools/accept_p8_1_observation_and_corner_phase.py` 的 Square 验证结束时，必须画并保存：

1) `square_v_ratio_cap.png`：v_ratio_cap(t)
2) `square_speed_util.png`：speed_util(t) 并画 y=1 参考线
（已有的 `square_v_ratio_exec.png`、`square_e_n.png` 继续保留）

> 图用 matplotlib 画即可；文件名固定，便于人类复核与版本对比。

## 3.2 新增强门禁断言（默认 Square）
补充以下断言（阈值可在文件顶部集中配置，便于调整）：

- 断言 A（速度非退化）：  
  - `v_ratio_exec_nunique >= 10`  
  - 否则报错：`"Degenerate speed: v_ratio_exec is (almost) constant. Check expert speed command or action mapping."`

- 断言 B（速度利用率不太差）：  
  - `mean_speed_util_in_band >= 0.5`  
  - 否则报错：`"Speed not utilized: expert not tracking v_cap/speed_target, or v_cap is too small everywhere."`

- 断言 C（v_cap 真的在“弯前下降”）：  
  - `std(v_ratio_cap) >= cap_std_min`（例如 0.02）  
  - 否则报错：`"v_ratio_cap not varying; braking envelope may be broken or dist_to_turn/turn_angle not wired."`

- 断言 D（跟随性）：  
  - `corr_v_exec_vs_v_cap >= 0.3`（正相关，阈值可调）  
  - 否则报错：`"v_exec does not track v_cap: check expert v_ratio_cmd computation or execution-layer filtering/clipping."`

> 这些门禁的意义：不让 Phase A 再出现“低速龟爬也算通过”的情况。

---

# 4) 是否需要人为验收？
需要，但只需要看 2 张新增图（成本极低）：

- `square_v_ratio_cap.png`：你要看到直线段 cap 高、拐角前 cap 下降（说明 braking envelope 在“算”）  
- `square_v_ratio_exec.png`：你要看到 exec 跟着 cap 有明显的升降（说明速度闭环“用起来了”）

如果两张图都符合直觉，同时 summary 门禁通过，就可以放行进入 Phase B。

> 反过来：如果门禁没通过，禁止进入 Phase B，先按报错提示定位（一般是 expert speed 命令没接 cap、或动作映射/滤波覆盖掉速度）。

---

# 5) 回归保护（必须）
1) 先跑默认 Square 的补丁门禁通过。  
2) 再跑 Line 与 S-shape（可只要求“不 NaN/Inf、能推进、不过早 stall”）。  
3) 最后再跑一遍本补丁 DoD（0.1）确保没有破坏既有脚本。

---

# 6) 最常见失败原因与快速定位（给 codex 自查）
- `v_ratio_exec_nunique` 仍然很小：  
  - 90% 是 expert 仍在用常数速度，或 v_ratio_cmd 写错列，或 action mapping 把 v_ratio 压死。
- `std(v_ratio_cap)` 很小：  
  - 说明 dist_to_turn/turn_angle 没接上，或 `_scan_for_next_turn` 没被 `_compute_p4_pre_step_status` 复用，或 corner 信息全为 0。
- `corr_v_exec_vs_v_cap` 很低：  
  - 说明执行层（加速度限制/滤波/裁剪）太强，或 v_cmd_after_clip 没跟 v_ratio_cmd 走。

---

**执行顺序（强制）：**
1) 先改 expert speed command（2.1）  
2) 再加 trace/summary 字段（2.2、2.3）  
3) 再加门禁断言与图输出（3.1、3.2）  
4) 运行 DoD（0.1）  
5) 展示两张新增图给用户 → 用户通过后才进入 Phase B
