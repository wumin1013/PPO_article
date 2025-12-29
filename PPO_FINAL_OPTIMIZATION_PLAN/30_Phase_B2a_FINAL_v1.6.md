# Phase B2a：拐角平滑（学习驱动）（FINAL v1.6）
版本日期：2025-12-30  
依赖：B1 已通过（7维或 Plan B）。

---

## 目标
- 相对 P0：拐角段平滑性显著改善
- 相对 P0：直线不退化
- 全程硬约束满足；KCM 零违规

---

## 执行要点
- 使用 A-1 固化的 corner_mask 与平滑指标，避免“统计噪声假提升”
- 训练：先少量 seed 看趋势，再扩展到 ≥5 seed（自动化）
- 产物必须包含：corner zoom + 平滑性曲线/统计

---

## 验收（PASS/FAIL）
### 硬约束
同目标文档硬约束全部满足。

### 直线不退化（相对 P0）
- `mean_v_ratio_straight >= baseline - 0.02`

### 平滑性必须改善（满足其一）
- `peak_jerk_corner <= baseline * 0.90`
- 或 `peak_ang_acc_corner <= baseline * 0.90`
- 或 `roughness_proxy <= baseline * 0.92`

