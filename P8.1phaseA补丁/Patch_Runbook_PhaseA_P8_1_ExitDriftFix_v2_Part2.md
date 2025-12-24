# Patch Runbook：PhaseA P8.1 ExitDriftFix v2 - Part2（Turn Completion + ang cap 修复）

> 目的：修复“出弯后状态不干净”和“ang cap 滞留/卡住”，确保直线段不再长期受限。本分册覆盖 Patch-2 与 Patch-3。

---

## 1) 你现在失败的“定量事实”

当前自动化 gate 失败仍然是：
- `max_abs_e_n=0.7717809 > 0.98*half_eps=0.735`
- `reached_target=false`
- `steps=72`
- `cap_ang_active_ratio=0.4861` 明显大于 `corner_mode_ratio=0.0972`

这些数据说明：
1) ang cap 在非拐角阶段仍大量生效。  
2) 出弯后的状态或 cap 逻辑存在滞留。

---

## 2) 根因判断（聚焦 turn completion 与 cap）

### 根因 B：turn completion/turn phase 内部状态不干净
拐角结束了，但某个用于 cap 计算的状态仍认为“还在 turn/还在高曲率预瞄区”。

### 根因 C：恢复限速/放行限速打在了错误的分量上
你希望 recovery/exit-rate-limit 作用在 **final cap**，但实际更像把恢复逻辑写进了 `v_ratio_cap_ang`，导致直线段也无法恢复到“正常直行”的速度/控制分布。

---

## 3) 设计原则（本分册）

- **turn completion 可证伪**：turn progress 只累计到 `turn_angle`，不允许跑到 `2*pi`。
- **恢复/放行只作用于 final cap**：ang/brake 分量保留物理含义。
- **诊断要能阻止盲调**：cap 指标异常直接 fail 并打印关键统计。

---

## 4) Patch-2（必做）：turn completion 变量按 turn_angle 截断 + 复位

### 4.1 位置
专家策略里维护 turn progress 的那段（你现在已经在 trace 里打了 `omega_ratio_cmd`，所以基本能定位到 turn 状态更新处）。

### 4.2 改法
- 引入 `theta_prog`（转弯进度），每步更新：
  - `theta_prog = clamp(theta_prog + omega_cmd * dt, 0, abs(turn_angle))`
- 当 `theta_prog >= abs(turn_angle)`：
  - `turn_done=True`
  - 强制进入 exit-settle：`omega_ratio_cmd` 采用直线闭环（而不是继续维持 corner 模式）
  - `theta_prog=0` 并且 **复位**任何“turn 累计量/角度 wrap 量”

**验收要点**：turn_angle 是 `pi/2` 时，theta_prog 不允许跑到 `2*pi`。

---

## 5) Patch-3（必做）：修 ang cap 的“滞留/卡住”（只作用于 final cap）

### 5.1 位置
`cnc_env.py`：你现在生成 `v_ratio_cap_brake / v_ratio_cap_ang / v_ratio_cap_final` 的同一处。

### 5.2 改法（非常具体）
1) 保留物理分量：
   - `v_ratio_cap_ang_raw`：只由“预瞄曲率/角速度可行性”算出
   - `v_ratio_cap_brake_raw`：只由刹车/减速度可行性算出
2) 再计算恢复/放行：
   - `v_ratio_cap_recovery`：只由 `abs(e_n)`、corner_exit settle 等逻辑算出
3) **最终**：
   - `v_ratio_cap_final = min(v_ratio_cap_ang_raw, v_ratio_cap_brake_raw, v_ratio_cap_recovery)`

**禁止**把 recovery 直接写进 `v_ratio_cap_ang`（否则会看到 cap_ang_active_ratio 畸形上升）。

### 5.3 新增一个 gate 诊断断言（强烈建议）
在 `accept_p8_1_observation_and_corner_phase.py` 里加：
- 统计 `cap_ang_active_ratio` 与 `corner_mode_ratio`
- 如果 `cap_ang_active_ratio > corner_mode_ratio + 0.2`：
  - 输出 “ang cap likely stuck on straight” + 打印 `mean(v_ratio_cap_ang_raw on straight)`
  - 直接 fail（避免反复盲调）

---

## 6) 验证与自验证

### 自动化必须通过
- `check_physics_logic.py` PASS
- `accept_p7_0_dynamics_and_scale.py` PASS
- `accept_p8_1_observation_and_corner_phase.py` PASS（包括 `reached_target=True`）

### 数值门槛（硬门槛）
- `max_abs_e_n ≤ 0.98*half_eps (=0.735)`
- `cap_ang_active_ratio <= corner_mode_ratio + 0.2`
- `mean_speed_util_in_band ≥ 0.85`

### 图表/人工复核（自验证）
- `square_v_cap_breakdown.png`：ang cap 主要在 corner 附近显著，直线段回归由 brake cap 或 1.0 主导
- `trace.csv`：直线段 `omega_ratio_cmd` 与 `cap` 逻辑正常切换，无长期滞留

---

## 7) 回滚与开关（避免越改越乱）

建议加 feature flags：
- `TURN_COMPLETION_CLAMP_ENABLE`
- `CAP_RECOVERY_APPLY_TO_FINAL_ONLY`

任一开关关闭，即可回到原有 turn/cap 逻辑。