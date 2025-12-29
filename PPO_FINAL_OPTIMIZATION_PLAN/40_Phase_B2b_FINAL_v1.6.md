# Phase B2b：出弯回线 + 软约束（FINAL v1.6）
版本日期：2025-12-30  
依赖：B2a 已通过。

---

## 目标
- exit_recovery 改善（出弯更快回到贴线）
- exit_oscillation 不恶化（不引入更强振荡）
- 直线不退化；硬约束仍满足

---

## 执行要点
- 使用 A-1 固化的 exit_active 与出弯指标
- 与 B2a 对比出弯指标趋势；与 P0 对比直线不退化

---

## 验收（PASS/FAIL）
### 硬约束
同目标文档硬约束全部满足。

### 出弯专项（相对 B2a）
- `exit_recovery_steps_mean` 不更差（最好更小）
- `exit_oscillation_rms` 不系统性上升

### 直线不退化（相对 P0）
- `mean_v_ratio_straight >= p0 - 0.02`

