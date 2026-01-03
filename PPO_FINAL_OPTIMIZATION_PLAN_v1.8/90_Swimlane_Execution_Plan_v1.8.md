# PPO 项目执行泳道图（FINAL v1.8）
版本日期：2026-01-03

> 泳道图目标：把你的工作拆成 4 条并行泳道：  
> **研究者（你） / 工程实现（Codex） / 验收与证据（工具链） / 论文产物（tex）**  
> 每完成一个 Phase，都必须触发“自动生成论文图表”。

---

## 1) 总览（Mermaid 泳道图）
```mermaid
flowchart LR
  subgraph L1[研究者：目标与实验设计]
    A[固定 baseline_ref=P0_L2] --> B[写 Hypothesis（1句）]
    B --> C[选 Phase：B1/B2a/B2b/B2c/D]
    C --> D[定义判定口径与阈值（Objectives）]
  end

  subgraph L2[工程实现：Codex]
    E[按 Phase 文档执行修改流程
（不做无关重构）] --> F[产出可运行版本]
  end

  subgraph L3[验收与证据：流水线]
    G[Train] --> H[Eval 生成 summary/trace]
    H --> I[A1 固化 Run Bundle]
    I --> J[A3 聚合 main_table/ablation_table]
  end

  subgraph L4[论文产物：自动生成]
    K[Phase C：生成 paper_assets
figures/tables] --> L[tex 引用 paper_assets]
    L --> M[一键重建论文图表]
  end

  D --> E
  F --> G
  J --> K
  K --> B
```

---

## 2) 里程碑顺序（按“目标逻辑”）
1) **B1（B0）**：观测口径/分段指标/plotter 对齐 → 生成 baseline 图包  
2) **B2a**：拐角平滑（至少 1 个不退化且改善的 run）  
3) **B2b**：出弯回线（解决慢漂）  
4) **B2c**：直线效率（速度提升）  
5) **C**：论文图表自动化（paper_assets 一键生成）  
6) **D**：消融与对比（tab_ablation + fig_ablation_vis）

---

## 3) 每个 Phase 的“完成条件”（一句话）
- **B1**：能自动计算分段指标 + P0_L2 一键出图  
- **B2a/b/c**：main_table 新增行 + 4 张对比图 + 1 页摘要  
- **C**：paper_assets 生成齐全且 tex 可直接引用  
- **D**：ablation_table 填满 + 自动生成消融图表

---
