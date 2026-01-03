# Phase B2c：直线效率（速度激励）（FINAL v1.8）
版本日期：2026-01-03  
依赖：B2a/B2b 至少已有 1 个“拐角更好且不退化”的候选 run。

---

## 1) 目标（直线更快，但贴线不退化）
相对 baseline_ref（P0_L2）以及 B2b 最优候选：
- non-corner 段 mean(v) 更高
- steps/cycle_time 更低（效率提升）
- max_abs_e_n 不退化、reached_target 不下降、硬约束 0

---

## 2) 交付物
- summary.json：non-corner 段速度统计 + 全程 steps/cycle_time
- plots：v(t) 对比（重点直线段）、overlay（确认贴线）
- main_table.csv：新增一行（含效率指标）

---

## 3) 修改流程（不写代码细节，只写步骤）
### Step 1：固定“直线段口径”
- non-corner 段定义来自 B1，保持不变

### Step 2：最小改动（一次只改 1 项）
- 只增强 non-corner 段的速度激励（或时间惩罚）
- 保持 corner_mask 段的平滑/回线逻辑不动（避免影响 B2a/B2b 的结论）

### Step 3：运行并归档
- 训练/评测 → A1 固化 → A3 聚合 → plotter 出图
- 输出 1 页摘要：效率指标 + v(t)/overlay

---

## 4) PASS/FAIL（建议）
PASS：
- 必须项不退化
- steps 明显下降或 non-corner mean(v) 上升
- 不出现“贴线变差换速度”的退化

FAIL：
- 必须项退化或效率未提升

---

## 5) 论文映射
- Fig：`fig:curve_s`（直线段 v(t) 对比更直观）
- Table：`tab:results` 的效率列（steps/cycle_time）应体现提升

---

## 6) 自动论文产物生成（B2c 完成后必做）
- 自动更新：`paper_assets/figures/v_profile_*`
- 自动更新：`paper_assets/tables/tab_results.csv`（效率列）

---
