# PPO_FINAL_OPTIMIZATION_PLAN（发送给 Codex 的执行包）FINAL v1.8
版本日期：2026-01-03  
目标：把优化工作严格收敛到 **“目标 → 证据 → 论文产物”** 的闭环；避免“文档越来越多、离目标越来越远”。

---

## 0) 你现在的真实状态（先对齐口径）
- 已完成：P0（可用基线）、A0/A1/A2/A3（Run Bundle + 聚合表流水线）
- 未完成（接下来必须做）：B0（观测/口径对齐）、B2a/B2b/B2c（目标三连）、C/D（论文结果与消融）

> 关键：后续所有 Phase 的“通过”必须能在 **main_table.csv + 自动生成图** 中复现。

---

## 1) Codex 接入方式（强约束）
### 1.1 允许做什么
- 只做：**按本文档的“修改流程”完成工程改造**（观测字段、评测统计、绘图自动化、配置组织）
- 必须：保持 P0_L2 作为 baseline_ref，任何改动都不允许破坏 P0 可复现性

### 1.2 禁止做什么
- 禁止一次性大重构（尤其是把训练语义改乱）
- 禁止“代码改完但不落盘、不验收、不出图”（这在科研上等价于没做）

---

## 2) 文件与执行顺序（必须按此顺序）
1) `01_Objectives_FINAL_v1.8.md`（目标/指标/论文映射）
2) `02_Workflow_FINAL_v1.8.md`（总工作流 + 证据链）
3) `20_Phase_B1_FINAL_v1.8.md`（B0 + B1：观测口径与可计算指标）
4) `30_Phase_B2a_FINAL_v1.8.md`（拐角平滑）
5) `40_Phase_B2b_FINAL_v1.8.md`（出弯回线）
6) `50_Phase_B2c_FINAL_v1.8.md`（直线效率）
7) `60_Phase_C_FINAL_v1.8.md`（论文图表自动生成）
8) `70_Phase_D_FINAL_v1.8.md`（消融与对比，论文表格闭环）
9) `90_Swimlane_Execution_Plan_v1.8.md`（泳道图总执行方案）

---

## 3) baseline_ref（唯一锚点）
- baseline_ref 固定指向：`artifacts/P0_L2/<P0_L2_BUNDLE>/`
- 所有 PhaseB 的 main_table/plot 必须包含 baseline 与当前 run 的对比输出。

---

## 4) 交付验收：如何判断“完成”
- 每个 Phase 的交付物必须是 **Run Bundle + main_table 新增行 + 自动生成论文图/表**
- 如果某 Phase 只改了代码但无法一键复现实验产物 → 视为未完成

---
