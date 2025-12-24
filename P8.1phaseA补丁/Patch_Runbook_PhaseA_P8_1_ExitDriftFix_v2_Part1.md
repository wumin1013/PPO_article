# Patch Runbook：PhaseA P8.1 ExitDriftFix v2 - Part1（直线段纠偏 + 恢复）

> 目的：把 `accept_p8_1_observation_and_corner_phase.py` 从 `max_abs_e_n=0.7718` 拉回到门槛以内（≤0.735），并让 `reached_target=True`。本分册仅覆盖 Patch-1（直线段纠偏 + 恢复模式）。

---

## 1) 你现在失败的“定量事实”

当前自动化 gate 失败仍然是：
- `max_abs_e_n=0.7717809 > 0.98*half_eps=0.735`
- `reached_target=false`
- `steps=72`

这些数据说明：
1) **越界幅度不大**（只超阈约 0.0368），属于“差一口气”的系统性漂移，而非一次性爆炸。  
2) 直线段纠偏不足导致“慢漂移 + 末端过冲”的概率较高。

---

## 2) 根因判断（聚焦直线段闭环）

### 根因 A：出弯后“直线段横向纠偏权能不足（或被死区/限幅吞掉）”
你给专家策略加了“硬归零窗口”，但 **e_n 从小负值开始持续变大**，说明直线段的纠偏不是稳定闭环（要么死区太大、要么 omega 限幅太小、要么只看 e_n 不看航向误差）。
典型表现就是：前段不纠偏（omega_cmd≈0），等误差积累到一定程度才突然饱和纠偏（omega_cmd 直接打满），于是出现“慢漂移 + 末端过冲”。

---

## 3) 设计原则（本分册）

- **不再用“硬归零窗口”当主要手段**：窗口只能应急，不是稳定控制律。你需要一个在直线段稳定收敛的闭环控制。
- **控制律简洁可调**：优先线性闭环 + 软限幅，避免隐藏死区。
- **恢复模式只在必要时触发**：通过滞回阈值控制进入/退出，避免频繁抖动。

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

---

## 5) 验证与自验证

### 自动化必须通过
- `check_physics_logic.py` PASS
- `accept_p7_0_dynamics_and_scale.py` PASS
- `accept_p8_1_observation_and_corner_phase.py` PASS（包括 `reached_target=True`）

### 数值门槛（硬门槛）
- `max_abs_e_n ≤ 0.98*half_eps (=0.735)`
- `reached_target=True`

### 图表/人工复核（自验证）
- `square_e_n.png`：无越界尖峰/尾段慢漂移
- `trace.csv`：直线段 `omega_ratio_cmd` 随 e_n 连续变化、且在误差很小时仍保持微小纠偏（无死区）

---

## 6) 回滚与开关（避免越改越乱）

建议加 feature flags：
- `EXPERT_STRAIGHT_PD_ENABLE`
- `EXPERT_RECOVERY_MODE_ENABLE`

任一开关关闭，即可回到原有直线段控制逻辑。