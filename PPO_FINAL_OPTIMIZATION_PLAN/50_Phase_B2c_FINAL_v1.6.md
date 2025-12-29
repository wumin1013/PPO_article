# Phase B2c：速度激励 + 直线高效进给（FINAL v1.6）
版本日期：2025-12-30  
依赖：B2b 已通过。

---

## 目标
- 直线段显著提速
- 不破硬约束
- 不让拐角平滑性反弹
- KCM 干预不应系统性上升

---

## 执行要点
- 与 P0 对比直线速度提升
- 与 B2a 对比平滑性不反弹（建议用 baseline2）
- ≥5 seed 自动化

---

## 验收（PASS/FAIL）
### 硬约束
同目标文档硬约束全部满足。

### 直线提速（相对 P0，建议）
- `mean_v_ratio_straight >= p0 + 0.02`

### 平滑性不显著变差（相对 B2a，建议）
- `peak_ang_acc_corner <= b2a * 1.05`
- `peak_jerk_corner <= b2a * 1.05`

### 干预度（推荐）
- `mean_kcm_intervention` 不显著上升（若显著上升，判为风险项）

