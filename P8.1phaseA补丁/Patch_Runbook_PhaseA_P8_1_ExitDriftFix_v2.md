# Patch Runbook：PhaseA P8.1 仍越界的“出弯漂移”修复（ExitDriftFix v2）

> 目的：把 `accept_p8_1_observation_and_corner_phase.py` 从 **max_abs_e_n=0.7718** 拉回到门槛以内（≤0.735），并让 `reached_target=True`，满足进入 PhaseB 的“系统可学习/可控”前置条件。

---

## 1) 你现在失败的“定量事实”

当前自动化 gate 失败仍然是：
- `max_abs_e_n=0.7717809 > 0.98*half_eps=0.735`
- `reached_target=false`
- `steps=72`
- 关键异常之一：`cap_ang_active_ratio=0.4861` 明显大于 `corner_mode_ratio=0.0972`（角速度 cap 在非拐角阶段仍大量生效） fileciteturn6file0L4-L19

这些数据说明：
1) **越界幅度不大**（只超阈约 0.0368），属于“差一口气”的系统性漂移，而非一次性爆炸。  
2) **ang cap 生效比例异常**：如果你的设计意图是“只在拐角附近压住”，那 `cap_ang_active_ratio` 应该与 `corner_mode_ratio` 同量级；现在大幅偏离，意味着 **cap 放行逻辑仍有“卡住/滞留”** 的可能。 fileciteturn6file0L12-L16

---

## 2) 根因判断（结合你当前的 trace 行为）

> 这里的判断不是拍脑袋：它解释了“为什么你加了 corner exit 滞回/放行限速/恢复限速 + 专家策略硬归零窗口后，仍会慢慢漂出界”。

### 根因 A：出弯后“直线段横向纠偏权能不足（或被死区/限幅吞掉）”
你给专家策略加了“硬归零窗口”，但 **e_n 从小负值开始持续变大**，说明直线段的纠偏不是稳定闭环（要么死区太大、要么 omega 限幅太小、要么只看 e_n 不看航向误差）。  
典型表现就是：前段不纠偏（omega_cmd≈0），等误差积累到一定程度才突然饱和纠偏（omega_cmd 直接打满），于是出现“慢漂移 + 末端过冲”。

### 根因 B：turn completion/turn phase 的内部状态不干净（出弯退出条件被误触发/误延迟）
你的 `corner_mode_ratio` 只有 0.0972（约 7/72 步），但 **ang cap 却持续更久**。这通常是“拐角状态机/预瞄曲率/turn phase 变量”没同步复位：拐角结束了，但某个用于 cap 计算的状态仍认为“还在 turn/还在高曲率预瞄区”。 fileciteturn6file0L12-L16

### 根因 C：恢复限速/放行限速“打在了错误的分量上”
你希望 recovery/exit-rate-limit 作用在 **final cap**（或者 v_exec），但从现象看更像 **v_ratio_cap_ang 被长期压在很小的值**，导致直线段也无法恢复到“正常直行”的速度/控制分布（即便速度低不一定会漂，但这通常会恶化控制器的工作区间）。 fileciteturn6file0L14-L16

---

## 3) 这次补丁的设计原则（非常具体）

- **不再用“硬归零窗口”当主要手段**：窗口只能应急，不是稳定控制律。你需要一个在直线段稳定收敛的闭环控制。
- **把“cap 放行/恢复”严格限定在 final cap 上**：ang/brake 分量要保留“物理含义”，避免把“恢复限速”误塞进 `v_ratio_cap_ang` 造成统计指标畸形。
- **把 turn completion 做成可证伪**：turn progress 必须只累计到 `turn_angle`，而不是累计到 2π 或靠 wrap。

---

## 4) Patch-1（必做）：直线段稳定纠偏控制律（去死区 + 软限幅）

### 4.1 位置
`accept_p8_1_observation_and_corner_phase.py` 内专家策略（或你封装的 expert policy 模块）。

### 4.2 改法（核心）
把直线段角速度命令统一为：

- `omega_ratio_cmd = clip( k_e * e_n + k_psi * psi_err + k_i * integ_e , -omega_ratio_max, +omega_ratio_max )`

建议默认：
- `k_e = 1.2 ~ 2.0`（按你的单位调整）
- `k_psi = 0.6 ~ 1.2`
- `k_i = 0`（先不要积分；如果后面还有稳态偏置再加）
- `omega_ratio_max_straight = 0.35 ~ 0.6`（不能太小，否则会“漂”；也不能太大，否则会“抖”）

**关键要求：**
- **禁止死区**：不要在 `|e_n|<某阈值` 时强制 omega=0（这正是“慢漂移”的温床）。
- `psi_err` 必须来自参考路径切向/期望航向，不要用 “omega_exec 之类的累计量” 直接当航向。

### 4.3 加一个“直线段恢复模式”（把你差的那 0.0368 吃掉）
当 `abs(e_n) > e_recover_on` 时，进入 recovery：
- `e_recover_on = 0.55*half_eps`（=0.4125）
- `e_recover_off = 0.35*half_eps`（=0.2625）

recovery 期间：
- `omega_ratio_max = min(0.8, omega_ratio_max_straight*1.5)`（给纠偏更强权能）
- `v_ratio_cmd = min(v_ratio_cmd, v_recovery)`，建议 `v_recovery = 0.012 ~ 0.016`

这能显著降低“末端越界”的概率，并且属于 PhaseA 允许的“工程补丁”（因为 gate 本质上是系统可控性验证）。

---

## 5) Patch-2（必做）：turn completion 变量按 turn_angle 截断 + 复位

> 这一步是为了避免“出弯后状态不干净”，导致 cap/控制逻辑仍按 turn 处理，或者直线段的航向误差计算失真。

### 5.1 位置
专家策略里维护 turn progress 的那段（你现在已经在 trace 里打了 `omega_ratio_cmd`，所以基本能定位到 turn 状态更新处）。

### 5.2 改法
- 引入 `theta_prog`（转弯进度），每步更新：
  - `theta_prog = clamp(theta_prog + omega_cmd * dt, 0, abs(turn_angle))`
- 当 `theta_prog >= abs(turn_angle)`：
  - `turn_done=True`
  - 强制进入 exit-settle：`omega_ratio_cmd` 采用上面 4.2 的直线闭环（而不是继续维持“corner 模式”）
  - `theta_prog=0` 并且 **复位**任何“turn 累计量/角度 wrap 量”

**验收要点**：turn_angle 是 `pi/2` 时，theta_prog 不允许跑到 `2*pi`。

---

## 6) Patch-3（必做）：修 ang cap 的“滞留/卡住”（只作用于 final cap）

### 6.1 位置
`cnc_env.py`：你现在生成 `v_ratio_cap_brake / v_ratio_cap_ang / v_ratio_cap_final` 的同一处。

### 6.2 改法（非常具体）
1) 保留物理分量：
   - `v_ratio_cap_ang_raw`：只由“预瞄曲率/角速度可行性”算出  
   - `v_ratio_cap_brake_raw`：只由刹车/减速度可行性算出  
2) 再计算恢复/放行：
   - `v_ratio_cap_recovery`：只由 `abs(e_n)`、corner_exit settle 等逻辑算出  
3) **最终**：
   - `v_ratio_cap_final = min(v_ratio_cap_ang_raw, v_ratio_cap_brake_raw, v_ratio_cap_recovery)`

**禁止**把 recovery 直接写进 `v_ratio_cap_ang`（否则你会看到 cap_ang_active_ratio 畸形上升）。

### 6.3 新增一个 gate 诊断断言（强烈建议）
在 `accept_p8_1_observation_and_corner_phase.py` 里加：

- 统计 `cap_ang_active_ratio` 与 `corner_mode_ratio`
- 如果 `cap_ang_active_ratio > corner_mode_ratio + 0.2`：
  - 输出 “ang cap likely stuck on straight” + 打印 `mean(v_ratio_cap_ang_raw on straight)`
  - 直接 fail（这能避免你反复“盲调”）

你现在这个差值是 `0.486 - 0.097 = 0.389`，明显触发。 fileciteturn6file0L12-L16

---

## 7) 验收标准（通过才允许进入 PhaseB）

### 自动化必须通过
- `check_physics_logic.py` PASS
- `accept_p7_0_dynamics_and_scale.py` PASS
- `accept_p8_1_observation_and_corner_phase.py` PASS（包括 `reached_target=True`）

### 数值门槛（硬门槛）
- `max_abs_e_n ≤ 0.98*half_eps (=0.735)`（全程都要在界内）
- `cap_ang_active_ratio <= corner_mode_ratio + 0.2`（不再出现直线段长期 ang cap）
- `mean_speed_util_in_band ≥ 0.85`（别靠“爬行”过 gate；你当前 0.962 是够的） fileciteturn6file0L8-L16

### 图表/人工复核（软门槛）
- `square_e_n.png`：无越界尖峰/尾段慢漂移
- `square_v_cap_breakdown.png`：ang cap 主要在 corner 附近显著，直线段回归由 brake cap 或 1.0 主导
- `trace.csv`：直线段 `omega_ratio_cmd` 随 e_n 连续变化、并能在误差很小时保持微小纠偏（而不是死区=0）

---

## 8) 回滚与开关（避免越改越乱）

建议加 feature flags：
- `EXPERT_STRAIGHT_PD_ENABLE`
- `EXPERT_RECOVERY_MODE_ENABLE`
- `TURN_COMPLETION_CLAMP_ENABLE`
- `CAP_RECOVERY_APPLY_TO_FINAL_ONLY`

每个 patch 都可以单独打开/关闭，方便定位哪个改动真正起作用。

---

## 9) 为什么这个补丁“更可能一把过”

你现在离门槛只差 0.0368，而且速度利用率已经很高。fileciteturn6file0L4-L16  
这说明环境/标定基本能跑通，剩下是：
- 直线段闭环要稳定收敛（去死区）
- turn completion 状态要干净（别把 90° 当 360°）
- recovery/exit-limiter 要作用在 final cap（别污染 ang 分量）

这三件事做对，P8.1 通过的概率会非常高。

