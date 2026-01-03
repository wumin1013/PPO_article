# Phase D：消融与对比（FINAL v1.8）
版本日期：2026-01-03  
依赖：C 已完成（能自动出图/出表）。

> 目标：回答审稿人最爱问的两句：  
> 1) 你到底比谁强？  
> 2) 强在哪里？为什么强？

---

## 1) D 的实验矩阵（最小但够用）
### 1.1 对比基线（必须）
- Baseline 1：P0_L2（你的工程基线）
- Baseline 2：传统规划/规则（论文里的“串行式”或等价实现）  
  （若你项目里已有 baseline runner，则用它；否则先定义“可复现的传统基线”）

### 1.2 消融项（建议最少 4 个开关）
- 去掉 corner 平滑项（验证 B2a 的贡献）
- 去掉 recovery 增强（验证 B2b 的贡献）
- 去掉 non-corner 速度激励（验证 B2c 的贡献）
- 去掉/弱化 KCM 或 shielding（仅在安全可控情况下做；否则改为统计对比）

---

## 2) 交付物
- ablation_table.csv：每个开关组合一行
- fig_ablation_vis：消融可视化（雷达图/条形图/矩阵图任选，但要可复现）
- tab_results：包含对比基线的量化对比
- Run Bundle：每个消融 run 都必须归档（否则论文不可复现）

---

## 3) 修改流程（不写代码细节，只写步骤）
### Step 1：定义实验矩阵（写入文档）
- 列出每个开关组合（A/B/C/D）
- 固定测试路径集合与 seeds（避免 cherry-pick）

### Step 2：批量运行（调用已有流水线）
- 每个组合：train/eval → A1 归档 → A3 聚合 → C 出图出表

### Step 3：结果审查（学术化判定）
- 只要出现“改善来自退化别的指标”，就不能作为主结果
- 报告均值 + 方差（至少 3 seeds）

---

## 4) 论文映射（D 就是你 tab:ablation 的数据源）
- Table：`tab:ablation`
- Figure：`fig:ablation_vis`
- 文字：解释“组件贡献”与“约束保障”

---

## 5) 自动论文产物生成（D 完成后必做）
- 自动更新：`paper_assets/tables/tab_ablation.csv`
- 自动更新：`paper_assets/figures/fig_ablation_vis.*`
- 在每个 bundle manifest 中记录 `ablation_key`（用于表格自动分组）

---
