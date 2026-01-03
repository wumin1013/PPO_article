# PPO_project 总工作流（FINAL v1.8）
版本日期：2026-01-03

> 目标：把“优化过程”固定成可复制的科研流水线：  
> **假设 → 最小改动 → 训练/评测 → 归档 → 聚合 → 出图 → 论文更新**

---

## 1) 固定循环（每个实验只做这 7 步）
1) **Hypothesis（一句话）**：这次要改善什么？为什么？
2) **最小改动**：一次只动一个旋钮（reward / obs / config 之一）
3) **Train**：产出 checkpoints/logs
4) **Eval**：产出 summary + trace
5) **A1 Run Bundle 固化**：写清 baseline_ref、改动点、verdict
6) **A3 聚合**：main_table / ablation_table 自动追加一行
7) **Paper Artifacts 自动出图/出表**：产出论文图/表素材（不美化）

---

## 2) 阶段推进顺序（按目标逻辑）
- **B0（观测/口径对齐）**：让拐角平滑/降速/出弯回线变成“可计算指标”
- **B2a（拐角平滑）**：论文创新点第一刀
- **B2b（出弯回线）**：解决慢漂/回不来
- **B2c（直线效率）**：直线更快但贴线不退化
- **C（论文图表自动化）**：所有 run 一键生成论文素材
- **D（消融与对比）**：把“为什么更好”写成证据

---

## 3) Stop Rule（防止瞎跑）
- 连续 2 个 run 在必须项 FAIL：暂停 PhaseB，回到 B0 查口径/判定/统计
- 出现“拐角改善但直线退化”的 run：不得进入下一阶段，先把奖励隔离（corner vs non-corner）

---

## 4) 每次你需要交付/留档的最小材料
- 本次 config.yaml（或参数快照）
- Run Bundle（manifest/summary/trace/plots）
- main_table.csv（更新后）
- 一张 P0 vs 当前 run 的对比图（overlay 或 v/e_n）

---
