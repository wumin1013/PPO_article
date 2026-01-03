# Phase B2a：拐角平滑（学习驱动）（FINAL v1.8）
版本日期：2026-01-03  
依赖：B1 已通过（分段统计/plotter/主表列齐全）。

---

## 1) 目标（论文最在意的那一部分）
相对 baseline_ref（P0_L2）：
- **拐角段平滑性显著改善**（可计算指标）
- **直线段不退化**（贴线与效率不劣化）
- **硬约束零违规**（统计为 0 或机制保证 + 说明）

---

## 2) 关键原则：平滑必须“只在拐角生效、可归因”
- 平滑奖励/正则只在 corner_mask 段生效
- 直线段奖励与控制逻辑尽量不动（否则不可归因）
- 每次实验只改变一个旋钮（例如只加一项 jerk_proxy 惩罚）

---

## 3) 交付物（每个 B2a run 都必须产出）
- Run Bundle（manifest/summary/trace/plots）
- summary.json：corner_mask 段 smoothness 指标（peak/mean 等）+ v_drop 指标
- plots：至少 4 张对比图
  1) overlay（拐角附近放大）
  2) v(t)
  3) e_n(t)
  4) omega/domega（或 jerk_proxy）在 corner_mask 段的对比
- main_table.csv：新增一行（含改善项）

---

## 4) 修改流程（不写代码细节，只写步骤）
### Step 1：选择“平滑主指标”
- 从 B1 的可用字段中选一个最稳定的：omega、domega、jerk_proxy
- 在文档中固定：统计方式（peak/mean/percentile）、统计窗口（corner_mask）

### Step 2：设计最小改动（一次只改 1 项）
- 只在 corner_mask 段引入平滑项（惩罚变化率或高频振荡）
- 若出现“龟速过弯”，再引入轻量速度下限保护（仍只在 corner_mask）

### Step 3：运行并归档
- 训练 → 评测 → A1 固化 → A3 聚合 → plotter 出图（全自动）
- 生成“B2a 对比小结”：P0 vs 当前 run 的 1 页摘要（指标 + 4 张图）

---

## 5) PASS/FAIL（建议）
PASS（必须项 + 至少 1 个改善项）：
- 必须项：reached_target 不变差；max_abs_e_n 与 steps 不劣化（容忍比例按 Objectives）
- 改善项：corner_mask 段 smoothness 指标显著下降（你可用 10~20% 作为起步阈值）
- 负面保护：不得通过降低全程速度来“假平滑”

FAIL：
- 任何必须项退化，或平滑指标未改善

---

## 6) 论文映射（B2a 直接对应结果图表）
- Fig：`fig:curve_s` / `fig:curve_butterfly`（S形/蝶形路径对比）
- Table：`tab:results`（加入 smoothness 与 v_drop 列）
- 文字：解释“为什么更顺”（用 corner_mask 分段曲线与统计）

---

## 7) 自动论文产物生成（B2a 完成后必做）
- 自动更新：`paper_assets/figures/`（S形/蝶形：overlay、v(t)、e_n(t)、omega/domega）
- 自动更新：`paper_assets/tables/tab_results.csv`（从 main_table 导出）
- 在 Run Bundle 的 manifest 中记录对应论文标签：`paper_targets=["fig:curve_s","fig:curve_butterfly","tab:results"]`

---

## 8) 一键清理（可选）
仅保留：P0_L2 基线包 + aggregation + B2a 最优 Run Bundle。

前提：已完成 A3 聚合并更新 `main_table.csv`。

```bash
python PPO_project/tools/cleanup_keep_best.py --phases B2a --apply
```
