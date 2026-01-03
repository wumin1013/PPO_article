# Phase B1：B0 + 观测口径与可计算指标（FINAL v1.8）
版本日期：2026-01-03  
依赖：A0/A1/A2/A3 已通过；baseline_ref 已固定为 P0_L2。

> 说明：原 v1.7.2 的 B1 同时承载“7维观测/Plan B”等内容。  
> 在 v1.8 中，B1 的第一使命是：**把 PhaseB 变成“可度量科研问题”。**

---

## 1) 目标（必须完成，否则后续都在黑箱里调参）
让每个 Run Bundle 的 `summary.json` 都能自动生成以下分段统计：
- **corner_mask 段**：平滑（omega/domega/jerk_proxy）、降速（v_drop_ratio 等）
- **corner_end 后窗口**：回线（误差收敛、带外步数）
- **non-corner 段**：直线效率（mean(v)、steps）

并保证这些统计可以稳定对比 baseline_ref（P0_L2）。

---

## 2) 交付物（完成 B1 就该看到这些）
- trace.csv：包含 corner_mask / dist_to_corner / omega/proxy / mode/proxy 等字段
- summary.json：包含上述分段指标（per-episode + aggregate）
- plots/：P0_L2 vs 当前 run 的自动对比图（至少 3 张：overlay / v(t) / e_n(t)）
- main_table.csv：新增列并能写入这些指标（即使空值也要有列）

---

## 3) 修改流程（不写代码细节，只写步骤）
### Step 1：定义“分段口径”（写入文档与 manifest）
- corner_mask 的判定规则（何时进入/退出拐角段）
- corner_end 后窗口定义（例如 N 步或固定时间）
- non-corner 定义（与 corner_mask 互补）

### Step 2：补齐 trace 字段（只增不改训练语义）
- 记录：corner_mask、dist_to_corner、corner_angle/curvature_proxy（若可得）
- 记录：omega、domega 或 jerk_proxy（取现有可稳定获取的量）
- 记录：mode/proxy（normal/corner/recovery）

### Step 3：让 summary 能自动计算分段指标
- 从 trace 计算并写入 summary：aggregate + per-episode
- 指标命名与 main_table 列名一一对应（避免后续手工对齐）

### Step 4：让 plotter 一键对比 baseline vs candidate
- 输入：两个 bundle
- 输出：overlay、v(t)、e_n(t)、omega/domega（若有）

---

## 4) 验收标准（B1 PASS）
- P0_L2 重新 eval 的必须项不退化（可达性/贴线/效率/硬约束）
- main_table.csv 能新增行，且分段指标不为空（至少 80% 列有值）
- plotter 对 P0_L2 能一键出图（这将作为后续所有 Phase 的“眼睛”）

---

## 5) 论文映射（完成 B1 后你就能写“指标定义与测试路径”）
- 对应 tex：`性能评价指标与测试路径`（指标定义口径固定）
- 为后续 Fig/Tab 提供可复现数据接口：`tab:results`、`tab:ablation` 的列口径

---

## 6) 自动论文产物生成（B1 完成后必做）
- 生成 **P0_L2 基线图包**（作为所有对比的 baseline）
- 生成 **指标定义表**（列名/单位/统计窗口），保存为 `paper_assets/metrics_spec.csv`

---
